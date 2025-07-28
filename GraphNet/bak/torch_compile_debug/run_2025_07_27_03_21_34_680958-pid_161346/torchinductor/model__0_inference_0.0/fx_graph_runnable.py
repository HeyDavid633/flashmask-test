
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5 8.0 8.6 9.0 10.0 12.0+PTX'
os.environ['TORCH_NCCL_USE_COMM_NONBLOCKING'] = '0'
os.environ['PYTORCH_VERSION'] = '2.7.0a0+79aa174'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.7.0a0+79aa174'
os.environ['NVIDIA_PYTORCH_VERSION'] = '25.04'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0a0+79aa17489c.nv25.04
# torch cuda version: 12.9
# torch git version: Unknown


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Apr__9_19:24:57_PDT_2025 
# Cuda compilation tools, release 12.9, V12.9.41 
# Build cuda_12.9.r12.9/compiler.35813241_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
        view = torch.ops.aten.view.default(arg0_1, [128, 768])
        mm = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1 = torch.ops.aten.view.default(mm, [1, 128, 2304]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        split = torch.ops.aten.split.Tensor(add, 768, -1);  add = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        view_2 = torch.ops.aten.view.default(getitem, [1, 128, 12, 64]);  getitem = None
        permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(getitem_1, [1, 128, 12, 64]);  getitem_1 = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(getitem_2, [1, 128, 12, 64]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        expand = torch.ops.aten.expand.default(permute, [1, 12, 128, 64]);  permute = None
        view_5 = torch.ops.aten.view.default(expand, [12, 128, 64]);  expand = None
        expand_1 = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 128]);  permute_3 = None
        view_6 = torch.ops.aten.view.default(expand_1, [12, 64, 128]);  expand_1 = None
        bmm = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm, [1, 12, 128, 128]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_7, 8.0);  view_7 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze);  unsqueeze = None
        mul = torch.ops.aten.mul.Tensor(sub, 10000.0);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(div, mul);  div = mul = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(sub_1, torch.float32);  sub_1 = None
        amax = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax);  convert_element_type_default = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_2 = torch.ops.aten.expand.default(convert_element_type_6, [1, 12, 128, 128]);  convert_element_type_6 = None
        view_8 = torch.ops.aten.view.default(expand_2, [12, 128, 128]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_2, [1, 12, 128, 64]);  permute_2 = None
        view_9 = torch.ops.aten.view.default(expand_3, [12, 128, 64]);  expand_3 = None
        bmm_1 = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
        view_10 = torch.ops.aten.view.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None
        permute_4 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11 = torch.ops.aten.view.default(clone, [1, 128, 768]);  clone = None
        view_12 = torch.ops.aten.view.default(view_11, [128, 768]);  view_11 = None
        mm_1 = torch.ops.aten.mm.default(view_12, arg4_1);  view_12 = arg4_1 = None
        view_13 = torch.ops.aten.view.default(mm_1, [1, 128, 768]);  mm_1 = None
        add_1 = torch.ops.aten.add.Tensor(view_13, arg5_1);  view_13 = arg5_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [2], correction = 0, keepdim = True)
        getitem_3 = var_mean[0]
        getitem_4 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_11, getitem_4);  convert_element_type_11 = getitem_4 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_12, [128, 768])
        mm_2 = torch.ops.aten.mm.default(view_14, arg8_1);  view_14 = arg8_1 = None
        view_15 = torch.ops.aten.view.default(mm_2, [1, 128, 3072]);  mm_2 = None
        add_5 = torch.ops.aten.add.Tensor(view_15, arg9_1);  view_15 = arg9_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5)
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.7071067811865476);  convert_element_type_15 = None
        erf = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_3, add_6);  mul_3 = add_6 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_16, [128, 3072]);  convert_element_type_16 = None
        mm_3 = torch.ops.aten.mm.default(view_16, arg10_1);  view_16 = arg10_1 = None
        view_17 = torch.ops.aten.view.default(mm_3, [1, 128, 768]);  mm_3 = None
        add_7 = torch.ops.aten.add.Tensor(view_17, arg11_1);  view_17 = arg11_1 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_12);  add_7 = convert_element_type_12 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True)
        getitem_5 = var_mean_1[0]
        getitem_6 = var_mean_1[1];  var_mean_1 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_19, getitem_6);  convert_element_type_19 = getitem_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg13_1);  mul_6 = arg13_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_7, arg12_1);  mul_7 = arg12_1 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
        return (convert_element_type_20,)
        
def load_args(reader):
    buf0 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (1, 128, 768), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (768, 2304), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (2304,), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 128, 128), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (768, 768), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (768,), dtype=torch.float16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (768,), dtype=torch.float16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (768,), dtype=torch.float16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (768, 3072), dtype=torch.float16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (3072,), dtype=torch.float16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (3072, 768), dtype=torch.float16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (768,), dtype=torch.float16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (768,), dtype=torch.float16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (768,), dtype=torch.float16, is_leaf=True)  # arg13_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)