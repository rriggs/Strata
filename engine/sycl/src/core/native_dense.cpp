#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/fused_gr_q8.hpp"
#include "strata/core/geometry.hpp"
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_HC_Q8 without a bf16 copy: the bf16 views' scratch, one matrix per (device, class, slot).  Allocated by
// `load` (never on first use: a graph capture must not meet the allocation).
struct HcScratch {
    explicit HcScratch(const sycl::device& d) : dev(d) {}
    sycl::device dev;        // sycl::device has no default constructor
    void* buf[2][2] = {};    // [class][slot]
    sycl::event ev[2];       // [slot]: the prompt path's last reader of the slot
    bool ev_set[2] = {};
};
std::mutex g_hc_mu;
std::vector<HcScratch>* g_hc_table = nullptr;
HcScratch* hc_slot(const sycl::device& d) {
    if (g_hc_table == nullptr) return nullptr;
    for (auto& e : *g_hc_table)
        if (e.dev == d) return &e;
    return nullptr;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// The native dense matrices (one allocation each, ~300 of them) are carved out of a few big blocks. On an Arc A-series card
// (i915) every kernel launch costs the host time proportional to the number of live device allocations (12 us with none,
// 131 us with 300, 1.8 ms with 3000) and the kernels themselves run slower; the allocation size does not matter.
// STRATA_DENSE_SLAB_MIB=<n>: block size in MiB (default 256 under the i915 driver, 0 = one allocation per matrix, which is
// what every other driver does); matrices of half a block or more keep their own allocation.
struct DenseSlab {
    struct Block { void* base; size_t size; size_t used; };
    std::mutex mu;
    std::vector<Block> blocks;
};
DenseSlab& dense_slab() { static DenseSlab s; return s; }
size_t dense_slab_bytes() {
    static const size_t v = [] {
        if (const char* e = std::getenv("STRATA_DENSE_SLAB_MIB")) return (size_t) (std::atol(e) > 0 ? std::atol(e) : 0) << 20;
        // on by default where it is measured: an Arc A-series card under the i915 driver (A750: 4K-token prompt 112.6 ->
        // 185.6 tok/s, output bit-identical); other drivers and cards keep one allocation per matrix
        return strata::intel_gpu_driver() == "i915" ? ((size_t) 256 << 20) : (size_t) 0;
    }();
    return v;
}
bool dense_slab_owns(const void* p) {
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (const auto& b : s.blocks) if ((uintptr_t) p >= (uintptr_t) b.base && (uintptr_t) p < (uintptr_t) b.base + b.size) return true;
    return false;
}
void* dense_alloc(size_t bytes) {
    const size_t sb = dense_slab_bytes();
    const size_t need = (bytes + 255) & ~(size_t) 255;
    if (sb == 0 || need >= sb / 2) return strata::malloc_device_guarded(bytes, dpct::get_in_order_queue());
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (auto& b : s.blocks)   // first fit: a gap left by a big matrix is filled by a later small one
        if (b.size - b.used >= need) { void* p = (uint8_t*) b.base + b.used; b.used += need; return p; }
    void* base = strata::malloc_device_guarded(sb, dpct::get_in_order_queue(), "native dense block");
    if (base == nullptr) return nullptr;
    s.blocks.push_back({base, sb, need});
    return base;
}
void dense_slab_free_blocks() {
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (auto& b : s.blocks) { strata::device_alloc_unregister(b.base); sycl::free(b.base, dpct::get_in_order_queue()); }
    s.blocks.clear();
}
struct DeviceFree {
    void
    operator()(void *p) const {
        if (p && !dense_slab_owns(p)) sycl::free(p, dpct::get_in_order_queue());
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
// The block with the most unused room is copied into an allocation of exactly its used size (and freed): the VRAM the
// last, part-filled block would waste is the expert cache's (192 slots became 53 on an 8 GB card without this).
// Called before the matrices' pointers are published, so the Pending items are the only holders of them.
bool dense_slab_trim(std::vector<Pending>& pending) {
    DenseSlab& s = dense_slab();
    size_t at = s.blocks.size(), best = 0;
    for (size_t i = 0; i < s.blocks.size(); ++i)
        if (s.blocks[i].size - s.blocks[i].used > best) { best = s.blocks[i].size - s.blocks[i].used; at = i; }
    if (at == s.blocks.size() || best < ((size_t) 4 << 20)) return true;
    DenseSlab::Block old = s.blocks[at];
    void* fresh = strata::malloc_device_guarded(old.used, dpct::get_in_order_queue(), "native dense block");
    if (fresh == nullptr) return true;   // keep the part-filled block
    dpct::get_in_order_queue().memcpy(fresh, old.base, old.used).wait();
    auto rebase = [&](DevicePtr& d) {
        const uintptr_t a = (uintptr_t) d.get();
        if (d && a >= (uintptr_t) old.base && a < (uintptr_t) old.base + old.used)
            (void) d.release(), d.reset((uint8_t*) fresh + (a - (uintptr_t) old.base));
    };
    for (auto& item : pending) { rebase(item.data); rebase(item.packed); }
    {
        std::lock_guard<std::mutex> lk(s.mu);
        s.blocks[at] = {fresh, old.used, old.used};   // the new block now owns those pointers
    }
    strata::device_alloc_unregister(old.base);
    sycl::free(old.base, dpct::get_in_order_queue());
    return true;
}
}

// the shared header declares the static member (b60aab1b); this port only needs the env switch
bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    for (void *p : weights_)
        if (!dense_slab_owns(p)) DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
    dense_slab_free_blocks();
}

bool NativeDense::load(const std::vector<std::string> &shards,
                       WeightTable &table, std::string &err,
                       bool include_ple_key, int64_t layer_lo,
                       int64_t layer_hi) try {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status =
                    DPCT_CHECK_ERROR(allocation = dense_alloc(bytes));
                DevicePtr data(allocation);
                if (status == 0)
                    status = DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue()
                            .memcpy(data.get(), gguf.tensor_data(tensor), bytes)
                            .wait());
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                if (status != 0) {
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    err = "native dense upload " + tensor.name + ": " +
                          dpct::get_error_string_dummy(status);
                        return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = DPCT_CHECK_ERROR(
                        packed_allocation = dense_alloc(bytes));
                    packed.reset(packed_allocation);
                    if (packed_status == 0)
                        packed_status = DPCT_CHECK_ERROR(
                            dpct::get_in_order_queue()
                                .memcpy(packed.get(), host.data(), bytes)
                                .wait());
                    /*
                    DPCT1000: Error handling if-stmt was detected but could
                    not be rewritten.
                    */
                    if (packed_status != 0) {
                        /*
                        DPCT1009: SYCL reports errors using exceptions and
                        does not use error codes. Please replace the
                        "get_error_string_dummy(...)" with a real error-handling
                        function.
                        */
                        /*
                        DPCT1001: The statement could not be removed.
                        */
                        err = "native dense packed upload " + tensor.name +
                              ": " +
                              dpct::get_error_string_dummy(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            // STRATA_HC_Q8=1: the hyper-connection down / up projections (the layers' and the final mixer's) as Q8_0 for
            // fused_gr_read_multi's Q8_0 read: the GGUF's own Q8_0 blocks (type 8, the Unsloth files), or Q8_0 of the
            // GGUF's BF16 values (type 30, the ISTA files; ggml's reference rounding, on the host).  Uploaded in the
            // read's split layout (fused_gr_q8_0_split_host).  The inject rows are not copied: the read takes the
            // pack's BF16 ones, so STRATA_HC_Q8_INJECT has no effect here.  Unless the BF16 copy is skipped
            // (add_hc_skip, the single-GPU default) the BF16 copies stay in the arena for the prompt path's GEMMs and
            // the Q8_0 copies are extra VRAM; with the skip they are the only copy and hc_bf16_view re-dequantizes.
            if (hc_q8_requested())
                for (const auto& tensor : gguf.tensors()) {
                    if ((tensor.type != 8 && tensor.type != 30) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2 ||
                        tensor.name.ends_with("_inject.weight") || !in_range(tensor.name)) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    // the read is compiled for the model geometry (hc_lr rows of hc * n_embd, or the transpose)
                    constexpr int64_t hc_dim = strata::core::kCompiledGeometry.hc_dim(), hc_lr = strata::core::kCompiledGeometry.hc_lr;
                    if (!((ref.ne0 == hc_dim && ref.ne1 == hc_lr) || (ref.ne0 == hc_lr && ref.ne1 == hc_dim))) {
                        err = "native dense (STRATA_HC_Q8): " + tensor.name + " is not a hyper-connection projection of the compiled geometry";
                        return false;
                    }
                    const uint64_t n = (uint64_t) ref.ne0 * (uint64_t) ref.ne1, bytes = n / 32 * 34;
                    std::vector<uint8_t> q8;
                    const uint8_t* blocks = nullptr;
                    if (ref.data == nullptr) {
                        // the BF16 copy was skipped (add_hc_skip): quantize the pack's row on the host
                        std::vector<uint16_t> host;
                        if (!table.read_bf16_host(tensor.name, host, err)) return false;
                        std::vector<float> f(host.size());
                        for (size_t i = 0; i < host.size(); ++i) {
                            const uint32_t u = (uint32_t) host[i] << 16;
                            std::memcpy(&f[i], &u, 4);
                        }
                        q8 = q8_0_of(f.data(), (uint64_t) f.size());
                        blocks = q8.data();
                    } else if (tensor.type == 30) {
                        const uint16_t* src = (const uint16_t*) gguf.tensor_data(tensor);
                        std::vector<float> f((size_t) n);
                        for (uint64_t i = 0; i < n; ++i) { const uint32_t u = (uint32_t) src[i] << 16; std::memcpy(&f[i], &u, 4); }
                        q8 = q8_0_of(f.data(), n);
                        blocks = q8.data();
                    } else {
                        blocks = (const uint8_t*) gguf.tensor_data(tensor);
                    }
                    std::vector<uint8_t> split((size_t) bytes);
                    strata::kernels::fused_gr_q8_0_split_host(blocks, split.data(), ref.ne1, ref.ne0);
                    void* p = nullptr;
                    if (DPCT_CHECK_ERROR(
                            p = (void *)strata::malloc_device_guarded(
                                bytes, dpct::get_in_order_queue())) != 0 ||
                        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(p, split.data(), bytes).wait()) != 0) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status =
            DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
                                 strata::kernels::native_q8_1_bytes(max_in),
                                 dpct::get_in_order_queue()));
        DevicePtr scratch(allocation);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            err = std::string("native dense scratch: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        dense_slab_trim(pending);
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        if (hc_q8_requested()) {
            // the scratches of hc_bf16_view (the rows without a bf16 copy), here and not on first use: a graph
            // capture must not meet the allocation
            int64_t elems = 0;
            for (const auto& kv : table.table_)
                if (kv.second.hc_q8 != nullptr && kv.second.data == nullptr)
                    elems = (std::max)(elems, (int64_t) kv.second.elements);
            if (elems != 0) {
                auto q = dpct::get_in_order_queue();
                std::lock_guard<std::mutex> lk(g_hc_mu);
                if (g_hc_table == nullptr) g_hc_table = new std::vector<HcScratch>();
                if (hc_slot(q.get_device()) == nullptr) {
                    HcScratch e(q.get_device());
                    for (int c = 0; c < 2; ++c)
                        for (int s = 0; s < 2; ++s) {
                            e.buf[c][s] = strata::malloc_device_guarded((size_t) elems * 2, q);
                            if (e.buf[c][s] == nullptr) { err = "native dense (STRATA_HC_Q8): scratch alloc"; return false; }
                            weights_.push_back(e.buf[c][s]);
                        }
                    g_hc_table->push_back(e);
                    std::fprintf(stderr, "strata: STRATA_HC_Q8=1 without the bf16 copies: %.1f MiB of bf16 view scratch\n",
                                 (double) elems * 2 * 4 / 1048576.0);
                }
            }
        }
        if (hc_q8_requested())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ---- STRATA_HC_Q8 without a bf16 copy (add_hc_skip): the CUDA engine's design, ported.  The projections live only as
// Q8_0 (the verify read's split layout); the prompt path's GEMMs and the layer read get a bf16 view of them, one
// matrix of scratch per (device, class, slot).  `cls` 0 = the prompt path (its streams share the pair, ordered by
// events: hc_bf16_release after the last GEMM that reads the slot), 1 = the layer / session read (its own pair: it is
// captured into graphs, where a wait on an outside event is illegal).
bool NativeDense::add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (!f) return false;
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        // "blk." only: the final mixer (output_hc_down / up) stays bf16 - it runs once a window
        if (n.rfind("blk.", 0) == 0 && hc_q8_name(n) && (n.ends_with("_down.weight") || n.ends_with("_up.weight"))) skip.insert(n);
    }
    std::fclose(f);
    return true;
}

bool NativeDense::gguf_carries_q8_hc(const std::vector<std::string>& shards) {
    try {
        for (const auto& path : shards) {
            const strata::GgufFile g(path);
            for (const auto& tensor : g.tensors())
                if (tensor.type == 8 && tensor.shape.size() == 2 && hc_q8_name(tensor.name) && !tensor.name.ends_with("_inject.weight")) return true;
        }
    } catch (const std::exception&) {}
    return false;
}

bool NativeDense::apply_hc_q8_default(const std::vector<std::string>& shards) {
    if (std::getenv("STRATA_HC_Q8") != nullptr || !gguf_carries_q8_hc(shards)) return false;
    setenv("STRATA_HC_Q8", "1", 0);
    return true;
}

const uint16_t* NativeDense::hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls) {
    if (w.data != nullptr) return (const uint16_t*) w.data;
    if (w.hc_q8 == nullptr) return nullptr;
    auto* q = strata::q_of(stream);
    if (q == nullptr || slot < 0 || slot > 1 || cls < 0 || cls > 1) return nullptr;
    HcScratch* e = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_hc_mu);
        e = hc_slot(q->get_device());
    }
    if (e == nullptr) return nullptr;
    void* buf = e->buf[cls][slot];
    if (buf == nullptr) return nullptr;
    // the prompt path: the scratch may still be read by a GEMM another stream enqueued (batch serving: several slots'
    // prompts).  The layer read (cls 1) never waits here: its own pair, and the wait would land in a captured graph.
    if (cls == 0 && e->ev_set[slot]) {
        const sycl::event dep = e->ev[slot];
        q->submit([&](sycl::handler& h) { h.depends_on(dep); });
    }
    strata::kernels::fused_gr_q8_0_split_to_bf16((const uint8_t*) w.hc_q8, (uint16_t*) buf, w.ne1, w.ne0, stream);
    return (const uint16_t*) buf;
}

void NativeDense::hc_bf16_release(void* stream, int slot, int cls) {
    if (cls != 0 || slot < 0 || slot > 1) return;
    auto* q = strata::q_of(stream);
    if (q == nullptr) return;
    HcScratch* e = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_hc_mu);
        e = hc_slot(q->get_device());
    }
    if (e == nullptr) return;
    e->ev[slot] = q->submit([&](sycl::handler& h) { h.host_task([] {}); });
    e->ev_set[slot] = true;
}

} // namespace strata::core
