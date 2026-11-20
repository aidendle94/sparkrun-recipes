"""Build `spark_csf_gemm_sm120`: FlashInfer's SM120 grouped NVFP4 GEMM with the weight operand's block scales read
from patches/scale_pack.py's packed form inside the kernel (no expansion buffer, nothing written to memory).

The stock kernel (CUTLASS `sm120_blockscaled_mma_array_tma.hpp`) loads four things a k tile with TMA: A, B and their
scale factors. With swap_ab the weights are operand A: a CTA's scale tile is 128 rows x TILE_K / 16 scales, 2,048
bytes for TILE_K = 256, and in the 128x4 layout those bytes are contiguous in memory and in the shared-memory stage in
the same order. scale_pack's groups are 32 consecutive bytes of that order, so a tile is 64 groups. Here the producer
warp writes the stage itself: each of its 32 lanes expands two groups (base + 4-bit offsets, or the raw group) from
global memory into the stage, then lane 0 issues the TMA loads of A, B and the activation scales as before, with the
stage's transaction byte count reduced by the scale tile. Consumers are unchanged. k tiles past the weight's width
get zero scales (TMA would have zero-filled them).

Everything is derived from the image's own sources by exact text replacement (a changed upstream line fails the
build instead of producing a wrong kernel): the CUTLASS header is shadowed through an include directory that comes
first, FlashInfer's wrapper .cuh is renamed and extended. One translation unit, two instantiations (BF16 and FP16
output), tile 128 x 64 x 256, swap_ab only. Hidden visibility and -Bsymbolic keep its template instantiations apart
from the stock gemm_sm120 module's in the same process.

    python3 build_csf_gemm.py OUT_DIR [--no-compile]      (inside the serving image; compiling needs no GPU)
writes OUT_DIR/gen/... and OUT_DIR/spark_csf_gemm_sm120.so. MIT License, Copyright (c) 2026 Aiden Le; the derived
sources keep their own licenses (CUTLASS BSD-3-Clause, FlashInfer Apache-2.0). What this script itself carries of
both (the lines it searches for, the rearranged producer loop, the binding source) is listed in the recipe's NOTICE,
with the licence texts it names.
"""
import os
import pathlib
import subprocess
import sys

NAME = "spark_csf_gemm_sm120"


def patch(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"build_csf_gemm: expected {count} of {old[:70]!r}..., found {text.count(old)}: the image's source changed")
    return text.replace(old, new)


DECODE = r'''
  // ---- spark: packed scale factors of operand A (patches/scale_pack.py), decoded by the producer warp ----
  static constexpr uint32_t CsfSFABytes = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>));
  static_assert(CsfSFABytes == size<0>(TileShape{}) * size<2>(TileShape{}) / 16,
                "the scale tile of operand A must be dense");
  static_assert(size<0>(TileShape{}) == 128 && CsfSFABytes % 512 == 0, "128-row tiles of whole 512-byte units");
  int32_t csf_group_{0};

  template <class T> static CUTLASS_DEVICE int csf_int(T const& t) { return static_cast<int>(cute::get<0>(t)); }
  static CUTLASS_DEVICE int csf_int(int t) { return t; }

  /// Every lane of the producer warp: expand this lane's groups of the (m_tile, k_tile) scale tile into the stage.
  CUTLASS_DEVICE void
  csf_decode(Params const& params, TensorStorage& shared_tensors, int m_tile, int k_tile, int write_stage) const {
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_tensors.smem_SFA.begin()) + size_t(write_stage) * CsfSFABytes;
    const size_t first = size_t(csf_group_) * size_t(params.csf_groups);
    const uint8_t* bases = params.csf_bases + first;
    const uint32_t* codes = reinterpret_cast<const uint32_t*>(params.csf_codes) + first * 4;
    constexpr int tile_groups = CsfSFABytes / 32;
    constexpr int tile_k4 = CsfSFABytes / 512;
#if SPARK_CSF_VARIANT == 9
    // floor: no scales at all (WRONG results; timing only)
    (void) stage; (void) bases; (void) codes; (void) m_tile; (void) k_tile;
#elif SPARK_CSF_VARIANT == 0
    for (int u = int(threadIdx.x & 31); u < tile_groups; u += 32) {
      uint64_t* out = reinterpret_cast<uint64_t*>(stage + u * 32);
      const int k4 = k_tile * tile_k4 + u / 16;
      if (k4 >= params.csf_k4) {
        out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
        continue;
      }
      const size_t g = (size_t(m_tile) * size_t(params.csf_k4) + size_t(k4)) * 16 + size_t(u % 16);
      const uint32_t base = bases[g];
      const uint32_t* c = codes + g * 4;
      if (base != 0xFFu) {
        const uint64_t spread = 0x0101010101010101ull * uint64_t(base);
        for (int w = 0; w < 4; ++w) {
          const uint32_t x = c[w];
          const uint64_t lo = x & 0x0F0F0F0Fu, hi = (x >> 4) & 0x0F0F0F0Fu;
          uint64_t v = 0;
          for (int j = 0; j < 4; ++j) {
            v |= (((lo >> (8 * j)) & 0xFFull) << (16 * j)) | (((hi >> (8 * j)) & 0xFFull) << (16 * j + 8));
          }
          out[w] = v + spread;
        }
      }
      else {
        const uint64_t* r = reinterpret_cast<const uint64_t*>(params.csf_raw) + size_t(c[0]) * 4;
        out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = r[3];
      }
    }
#else
    // variant 1 (the default): a lane's loads are issued together and each is one instruction (a 16-byte vector
    // load of a group's codes, a byte for its base): a load stalls the warp, so their number sets the producer's pace
    static_assert(tile_groups % 32 == 0 && tile_groups <= 64, "one or two groups a lane");
    constexpr int per_lane = tile_groups / 32;
    const int lane = int(threadIdx.x & 31);
    uint32_t base[2] = {0, 0};
    uint4 cv[2];
    bool live[2] = {false, false};
    for (int i = 0; i < per_lane; ++i) {
      const int u = lane + 32 * i;
      const int k4 = k_tile * tile_k4 + u / 16;
      live[i] = k4 < params.csf_k4;
      if (live[i]) {
        const size_t g = (size_t(m_tile) * size_t(params.csf_k4) + size_t(k4)) * 16 + size_t(u % 16);
        base[i] = bases[g];
        cv[i] = *reinterpret_cast<const uint4*>(codes + g * 4);
      }
    }
    for (int i = 0; i < per_lane; ++i) {
      ulonglong2* out = reinterpret_cast<ulonglong2*>(stage + (lane + 32 * i) * 32);
      if (!live[i]) {
        out[0] = make_ulonglong2(0, 0); out[1] = make_ulonglong2(0, 0);
      }
      else if (base[i] != 0xFFu) {
        const uint64_t spread = 0x0101010101010101ull * uint64_t(base[i]);
        const uint32_t c[4] = {cv[i].x, cv[i].y, cv[i].z, cv[i].w};
        uint64_t o[4];
        for (int w = 0; w < 4; ++w) {
          const uint64_t lo = c[w] & 0x0F0F0F0Fu, hi = (c[w] >> 4) & 0x0F0F0F0Fu;
          uint64_t v = 0;
          for (int j = 0; j < 4; ++j) {
            v |= (((lo >> (8 * j)) & 0xFFull) << (16 * j)) | (((hi >> (8 * j)) & 0xFFull) << (16 * j + 8));
          }
          o[w] = v + spread;
        }
        out[0] = make_ulonglong2(o[0], o[1]); out[1] = make_ulonglong2(o[2], o[3]);
      }
      else {
        const ulonglong2* r = reinterpret_cast<const ulonglong2*>(params.csf_raw) + size_t(cv[i].x) * 2;
        out[0] = r[0]; out[1] = r[1];
      }
    }
#endif
  }
'''

CSF_LOOP = r'''
    if (params.csf_bases != nullptr) {
      // spark: the stock setup of load(), then a loop every lane runs (lane 0 alone acquires the stage and issues TMA)
SETUP
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        if (lane_predicate) {
          pipeline.producer_acquire(smem_pipe_write);
        }
        __syncwarp();
        int write_stage = smem_pipe_write.index();
        csf_decode(params, shared_tensors, static_cast<int>(m_coord), csf_int(*k_tile_iter), write_stage);
        __syncwarp();
        if (lane_predicate) {
          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(params.tma_load_a.with(get<0>(input_tensormaps),*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
          copy(params.tma_load_b.with(get<1>(input_tensormaps),*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
          copy(params.tma_load_sfb.with(get<3>(input_tensormaps),*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));
        }
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
'''


def fork_header(text):
    text = patch(text, "#pragma once\n", "#pragma once\n// spark_csf_gemm: derived from CUTLASS's header by kernels/csf_gemm/build_csf_gemm.py (packed scale factors of A)\n")
    text = patch(text, "    ElementSF const** ptr_SFB{nullptr};\n    LayoutSFB layout_SFB{};\n  };\n",
                 "    ElementSF const** ptr_SFB{nullptr};\n    LayoutSFB layout_SFB{};\n"
                 "    const uint8_t* csf_bases{nullptr};\n    const uint8_t* csf_codes{nullptr};\n    const uint8_t* csf_raw{nullptr};\n"
                 "    int32_t csf_k4{0};\n    int32_t csf_groups{0};\n  };\n")
    text = patch(text, "    ElementSF const** ptr_SFB;\n    LayoutSFB layout_SFB;\n  };\n",
                 "    ElementSF const** ptr_SFB;\n    LayoutSFB layout_SFB;\n"
                 "    const uint8_t* csf_bases;\n    const uint8_t* csf_codes;\n    const uint8_t* csf_raw;\n"
                 "    int32_t csf_k4;\n    int32_t csf_groups;\n  };\n" + DECODE)
    text = patch(text, "      tma_load_sfb,\n      TmaTransactionBytes,\n      TmaTransactionBytesMK,\n      TmaTransactionBytesNK,\n",
                 "      tma_load_sfb,\n"
                 "      args.csf_bases != nullptr ? TmaTransactionBytes - CsfSFABytes : TmaTransactionBytes,\n"
                 "      args.csf_bases != nullptr ? TmaTransactionBytesMK - CsfSFABytes : TmaTransactionBytesMK,\n"
                 "      TmaTransactionBytesNK,\n")
    text = patch(text, "      reinterpret_cast<ElementSF const**>(args.ptr_SFB),\n      args.layout_SFB\n    };\n",
                 "      reinterpret_cast<ElementSF const**>(args.ptr_SFB),\n      args.layout_SFB,\n"
                 "      args.csf_bases,\n      args.csf_codes,\n      args.csf_raw,\n      args.csf_k4,\n      args.csf_groups\n    };\n")
    text = patch(text, "      int32_t next_batch) {\n    if (cute::elect_one_sync()) {\n      // Replacing global_address for the next batch\n",
                 "      int32_t next_batch) {\n    csf_group_ = next_batch;  // spark: every lane keeps the group it serves\n"
                 "    if (cute::elect_one_sync()) {\n      // Replacing global_address for the next batch\n")
    # load(): the stock block runs when the scales are not packed; a copy of its setup feeds the packed loop
    start = "    int lane_predicate = cute::elect_one_sync();\n\n    if (lane_predicate) {\n\n      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});"
    if text.count(start) != 1:
        raise SystemExit("build_csf_gemm: load()'s opening changed")
    at = text.index(start)
    setup_from = text.index("      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});", at)
    setup_to = text.index("      // Mainloop\n      CUTLASS_PRAGMA_NO_UNROLL\n      for ( ; k_tile_count > 0; --k_tile_count) {", setup_from)
    setup = text[setup_from:setup_to]
    if "tBsSFB" not in setup or "m_coord" not in setup:
        raise SystemExit("build_csf_gemm: load()'s setup changed")
    end = "        ++k_tile_iter;\n        ++smem_pipe_write;\n      }\n    }\n    __syncwarp();\n  }\n"
    if text.count(end) != 1 or text.index(end) < setup_to:
        raise SystemExit("build_csf_gemm: load()'s end changed")
    text = text.replace("    int lane_predicate = cute::elect_one_sync();\n\n    if (lane_predicate) {\n\n      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});",
                        "    int lane_predicate = cute::elect_one_sync();\n\n    if (params.csf_bases == nullptr && lane_predicate) {\n\n      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});", 1)
    text = text.replace(end, "        ++k_tile_iter;\n        ++smem_pipe_write;\n      }\n    }\n" + CSF_LOOP.replace("SETUP", setup) + "    __syncwarp();\n  }\n", 1)
    return text


def fork_cuh(text):
    for old, new in (("FLASHINFER_GROUP_GEMM_NVFP4_GROUPWISE_SM120_CUH_", "SPARK_GROUP_GEMM_NVFP4_CSF_SM120_CUH_"),
                     ("CutlassNVFP4GroupwiseScaledGroupGEMMSM120", "SparkCsfNVFP4GroupGEMMSM120"),
                     ("compute_sm120_cutlass_nvfp4_group_gemm_args", "spark_csf_compute_group_gemm_args"),
                     ("INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120", "INSTANTIATE_SPARK_CSF_GROUP_GEMM")):
        if old not in text:
            raise SystemExit(f"build_csf_gemm: {old} is not in FlashInfer's wrapper any more")
        text = text.replace(old, new)
    if '#include "../' not in text:
        raise SystemExit("build_csf_gemm: the wrapper's relative includes changed")
    text = text.replace('#include "../', '#include "flashinfer/')  # the wrapper moved out of flashinfer/gemm/
    # the packed planes ride behind the stream arguments of every signature and call (macro lines end in a
    # backslash after padding: the replacements keep whatever follows on the line)
    extra = "const uint8_t* csf_bases, const uint8_t* csf_codes, const uint8_t* csf_raw, int csf_k4, int csf_groups"
    text = patch(text, "    int device_id);", "    int device_id, " + extra + ");")
    text = patch(text, "          cudaStream_t stream, int device_id) {", "          cudaStream_t stream, int device_id, " + extra + ") {")
    text = patch(text, "      int device_id) {", "      int device_id, " + extra + ") {")
    text = patch(text, "            layout_SFB,", "            layout_SFB, SwapAB ? csf_bases : nullptr, csf_codes, csf_raw, csf_k4, csf_groups,")
    text = patch(text, "        SFB, D, alpha, m_indptr, n, k, num_groups, stream, device_id);",
                 "        SFB, D, alpha, m_indptr, n, k, num_groups, stream, device_id, csf_bases, csf_codes, csf_raw, csf_k4, csf_groups);")
    return text


SOURCE = r'''// spark_csf_gemm_sm120: written by kernels/csf_gemm/build_csf_gemm.py. Apache-2.0 (FlashInfer's wrapper) / MIT.
#include <spark/group_gemm_nvfp4_csf_sm120.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using namespace flashinfer::group_gemm;

namespace flashinfer {
namespace group_gemm {
INSTANTIATE_SPARK_CSF_GROUP_GEMM(128, 64, 256, true, cutlass::float_e2m1_t, cutlass::float_e2m1_t,
                                 cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::bfloat16_t, float_e2m1_t,
                                 float_e2m1_t, float_ue4m3_t, float_ue4m3_t, bfloat16_t)
INSTANTIATE_SPARK_CSF_GROUP_GEMM(128, 64, 256, true, cutlass::float_e2m1_t, cutlass::float_e2m1_t,
                                 cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::half_t, float_e2m1_t,
                                 float_e2m1_t, float_ue4m3_t, float_ue4m3_t, half_t)
}  // namespace group_gemm
}  // namespace flashinfer

// a (rows, k / 2) activations, b (groups, n, k / 2) weights, a_sf the activations' 128x4 scales; the weights' scales
// are either b_sf (stock layout, bases empty) or the packed planes bases (groups, G), codes (groups, G, 16), raw (R, 32).
void SparkGroupGemmNvfp4Csf(TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A,
                            TensorView B, TensorView SFA, TensorView SFB, TensorView D, TensorView alpha,
                            TensorView m_indptr, TensorView bases, TensorView codes, TensorView raw, int64_t n,
                            int64_t k, int64_t csf_k4, int64_t csf_groups) {
  int device_id = float_workspace_buffer.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  auto stream = get_stream(float_workspace_buffer.device());
  int num_groups = m_indptr.size(0) - 1;
  TVM_FFI_ICHECK(alpha.size(0) == num_groups || alpha.numel() == 0) << "alpha must have one value a group or be empty";
  const bool packed = bases.numel() != 0;
  const uint8_t* p_bases = packed ? static_cast<const uint8_t*>(bases.data_ptr()) : nullptr;
  const uint8_t* p_codes = packed ? static_cast<const uint8_t*>(codes.data_ptr()) : nullptr;
  const uint8_t* p_raw = packed ? static_cast<const uint8_t*>(raw.data_ptr()) : nullptr;
  // with packed scales the kernel never reads the weights' scale tensor: any valid address serves the descriptors
  auto* sfb = static_cast<cutlass::float_ue4m3_t*>(packed ? const_cast<void*>(static_cast<const void*>(p_bases)) : SFB.data_ptr());
  float* p_alpha = alpha.numel() == 0 ? nullptr : static_cast<float*>(alpha.data_ptr());
  void* ibuf = int_workspace_buffer.data_ptr();
  size_t isize = get_element_size(int_workspace_buffer) * int_workspace_buffer.size(0);
  void* fbuf = float_workspace_buffer.data_ptr();
  size_t fsize = get_element_size(float_workspace_buffer) * float_workspace_buffer.size(0);
  auto* a = static_cast<cutlass::float_e2m1_t*>(A.data_ptr());
  auto* b = static_cast<cutlass::float_e2m1_t*>(B.data_ptr());
  auto* sfa = static_cast<cutlass::float_ue4m3_t*>(SFA.data_ptr());
  int* indptr = static_cast<int*>(m_indptr.data_ptr());
  cudaError_t status;
  if (encode_dlpack_dtype(D.dtype()) == bfloat16_code) {
    status = SparkCsfNVFP4GroupGEMMSM120_128_64_256_true_float_e2m1_t_float_e2m1_t_float_ue4m3_t_float_ue4m3_t_bfloat16_t(
        ibuf, isize, fbuf, fsize, a, b, sfa, sfb, static_cast<cutlass::bfloat16_t*>(D.data_ptr()), p_alpha, indptr,
        static_cast<int>(n), static_cast<int>(k), num_groups, stream, device_id, p_bases, p_codes, p_raw,
        static_cast<int>(csf_k4), static_cast<int>(csf_groups));
  } else {
    TVM_FFI_ICHECK(encode_dlpack_dtype(D.dtype()) == float16_code) << "the output is BF16 or FP16";
    status = SparkCsfNVFP4GroupGEMMSM120_128_64_256_true_float_e2m1_t_float_e2m1_t_float_ue4m3_t_float_ue4m3_t_half_t(
        ibuf, isize, fbuf, fsize, a, b, sfa, sfb, static_cast<cutlass::half_t*>(D.data_ptr()), p_alpha, indptr,
        static_cast<int>(n), static_cast<int>(k), num_groups, stream, device_id, p_bases, p_codes, p_raw,
        static_cast<int>(csf_k4), static_cast<int>(csf_groups));
  }
  TVM_FFI_ICHECK(status == cudaSuccess) << "spark_csf_gemm failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_nvfp4_csf, SparkGroupGemmNvfp4Csf);
'''


def main():
    out = pathlib.Path(sys.argv[1]).resolve()
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.1a")
    variant = os.environ.get("SPARK_CSF_VARIANT", "1")  # 1 = the kernel that ships; 0 and 9 are for timing
    global NAME
    NAME = NAME if variant == "1" else f"{NAME}_v{variant}"
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(out / "ws")
    import flashinfer.jit.env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    inc = out / "gen" / "include"
    (inc / "cutlass/gemm/collective").mkdir(parents=True, exist_ok=True)
    (inc / "spark").mkdir(parents=True, exist_ok=True)
    stock_header = next(p / "cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp" for p in jit_env.CUTLASS_INCLUDE_DIRS
                        if (p / "cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp").is_file())
    (inc / "cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp").write_text(fork_header(stock_header.read_text()))
    stock_cuh = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh"
    (inc / "spark/group_gemm_nvfp4_csf_sm120.cuh").write_text(fork_cuh(stock_cuh.read_text()))
    source = out / "gen" / f"{NAME}.cu"
    source.write_text(SOURCE)
    print("sources written under", out / "gen")
    if "--no-compile" in sys.argv:
        return
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(NAME, [source], extra_include_paths=[inc, jit_env.FLASHINFER_CSRC_DIR],
                        extra_cuda_cflags=flags + ["-DCUTLASS_ENABLE_GDC_FOR_SM100=1", f"-DSPARK_CSF_VARIANT={variant}", "-Xcompiler", "-fvisibility=hidden",
                                                   "-Xcompiler", "-fvisibility-inlines-hidden"],
                        extra_cflags=["-fvisibility=hidden"], extra_ldflags=["-Wl,-Bsymbolic"])
    spec.build(verbose=True) if "verbose" in spec.build.__code__.co_varnames else spec.build()
    lib = spec.jit_library_path
    target = out / f"{NAME}.so"
    target.write_bytes(lib.read_bytes())
    print("built", target, target.stat().st_size, "bytes")
    print(subprocess.run(["sh", "-c", f"nm -D --defined-only {target} | grep -c ' T ' ; nm -D --defined-only {target} | grep -i group_gemm | head -3"],
                         capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
