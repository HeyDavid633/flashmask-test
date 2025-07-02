//  /flashmask-test/flash-attention/csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu

// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"


#include "include/namespace_config.h"
#include "include/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {


// flash_fwd_hdim64_fp16_sm80
template<>
void run_mha_fwd_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
}


// flash_fwd_hdim64_fp16_causal_sm80
template<>
void run_mha_fwd_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE