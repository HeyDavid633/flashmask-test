
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 9.0+PTX'
os.environ['PYTORCH_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['NVIDIA_PYTORCH_VERSION'] = '24.12'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root/tmpuqp_iipx'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_root/tmpuqp_iipx/triton'

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
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0+cu126
# torch cuda version: 12.6
# torch git version: 134179474539648ba7dee1317959529fbd0e7f89


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78):
        view = torch.ops.aten.view.default(primals_1, [-1, 128]);  primals_1 = None
        embedding = torch.ops.aten.embedding.default(primals_2, view);  primals_2 = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0)
        embedding_1 = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        view_1 = torch.ops.aten.view.default(primals_4, [1, -1]);  primals_4 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(view_1, torch.bool);  view_1 = None
        add_1 = torch.ops.aten.add.Tensor(iota, 0)
        iota_2 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_2 = torch.ops.aten.view.default(iota, [128, 1]);  iota = None
        le = torch.ops.aten.le.Tensor(add_1, view_2);  view_2 = None
        full_default = torch.ops.aten.full.default([128, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        view_4 = torch.ops.aten.view.default(iota_2, [1, 1]);  iota_2 = None
        index = torch.ops.aten.index.Tensor(convert_element_type, [view_4, add_1]);  convert_element_type = view_4 = add_1 = None
        view_5 = torch.ops.aten.view.default(index, [1, 1, 128]);  index = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, view_5);  bitwise_and = view_5 = None
        view_6 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 128, 128]);  bitwise_and_1 = None
        expand = torch.ops.aten.expand.default(view_6, [1, 1, 128, 128]);  view_6 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None
        view_7 = torch.ops.aten.view.default(convert_element_type_2, [-1, 768]);  convert_element_type_2 = None
        addmm = torch.ops.aten.addmm.default(primals_7, view_7, primals_8);  primals_7 = None
        view_8 = torch.ops.aten.view.default(addmm, [1, 128, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_8, 768, 2);  view_8 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_9 = torch.ops.aten.view.default(getitem_3, [1, 128, -1, 64]);  getitem_3 = None
        permute = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        view_10 = torch.ops.aten.view.default(getitem_4, [1, 128, -1, 64]);  getitem_4 = None
        permute_1 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        view_11 = torch.ops.aten.view.default(getitem_2, [1, 128, -1, 64]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        clone_2 = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        clone_3 = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  expand = full_default_2 = full_default_1 = None
        expand_1 = torch.ops.aten.expand.default(where, [1, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_1, clone_2, clone_3, expand_1, True)
        getitem_5 = _scaled_dot_product_efficient_attention[0]
        getitem_6 = _scaled_dot_product_efficient_attention[1]
        getitem_7 = _scaled_dot_product_efficient_attention[2]
        getitem_8 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_3 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_12 = torch.ops.aten.view.default(permute_3, [1, 128, -1]);  permute_3 = None
        view_13 = torch.ops.aten.view.default(view_12, [-1, 768]);  view_12 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view_13, primals_10);  primals_9 = view_13 = None
        view_14 = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
        add_4 = torch.ops.aten.add.Tensor(view_14, add);  view_14 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_9, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_1[0]
        getitem_10 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_9, getitem_10);  convert_element_type_9 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_11);  mul_2 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_6, torch.float16);  add_6 = None
        view_15 = torch.ops.aten.view.default(convert_element_type_10, [-1, 768]);  convert_element_type_10 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_13, view_15, primals_14);  primals_13 = None
        view_16 = torch.ops.aten.view.default(addmm_2, [1, 128, 3072])
        mul_4 = torch.ops.aten.mul.Tensor(view_16, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_16, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_16, mul_5);  view_16 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        view_17 = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_15, view_17, primals_16);  primals_15 = None
        view_18 = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
        add_9 = torch.ops.aten.add.Tensor(add_4, view_18);  view_18 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_2[0]
        getitem_12 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_12);  convert_element_type_17 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, primals_17);  mul_8 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, primals_18);  mul_9 = primals_18 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_11, torch.float16);  add_11 = None
        view_19 = torch.ops.aten.view.default(convert_element_type_18, [-1, 768]);  convert_element_type_18 = None
        addmm_4 = torch.ops.aten.addmm.default(primals_19, view_19, primals_20);  primals_19 = None
        view_20 = torch.ops.aten.view.default(addmm_4, [1, 128, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_20, 768, 2);  view_20 = None
        getitem_13 = split_1[0]
        getitem_14 = split_1[1]
        getitem_15 = split_1[2];  split_1 = None
        view_21 = torch.ops.aten.view.default(getitem_14, [1, 128, -1, 64]);  getitem_14 = None
        permute_4 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        view_22 = torch.ops.aten.view.default(getitem_15, [1, 128, -1, 64]);  getitem_15 = None
        permute_5 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        view_23 = torch.ops.aten.view.default(getitem_13, [1, 128, -1, 64]);  getitem_13 = None
        permute_6 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        clone_6 = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        clone_7 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        clone_8 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_6, clone_7, clone_8, expand_1, True)
        getitem_16 = _scaled_dot_product_efficient_attention_1[0]
        getitem_17 = _scaled_dot_product_efficient_attention_1[1]
        getitem_18 = _scaled_dot_product_efficient_attention_1[2]
        getitem_19 = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        permute_7 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        view_24 = torch.ops.aten.view.default(permute_7, [1, 128, -1]);  permute_7 = None
        view_25 = torch.ops.aten.view.default(view_24, [-1, 768]);  view_24 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_21, view_25, primals_22);  primals_21 = view_25 = None
        view_26 = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
        add_12 = torch.ops.aten.add.Tensor(view_26, add_9);  view_26 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_12, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_25, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_3[0]
        getitem_21 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_25, getitem_21);  convert_element_type_25 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, primals_23);  mul_10 = None
        add_14 = torch.ops.aten.add.Tensor(mul_11, primals_24);  mul_11 = primals_24 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None
        view_27 = torch.ops.aten.view.default(convert_element_type_26, [-1, 768]);  convert_element_type_26 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_25, view_27, primals_26);  primals_25 = None
        view_28 = torch.ops.aten.view.default(addmm_6, [1, 128, 3072])
        mul_12 = torch.ops.aten.mul.Tensor(view_28, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_28, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15 = torch.ops.aten.add.Tensor(view_28, mul_13);  view_28 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_29 = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_27, view_29, primals_28);  primals_27 = None
        view_30 = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
        add_17 = torch.ops.aten.add.Tensor(add_12, view_30);  view_30 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_4[0]
        getitem_23 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_23);  convert_element_type_33 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_29);  mul_16 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, primals_30);  mul_17 = primals_30 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None
        view_31 = torch.ops.aten.view.default(convert_element_type_34, [-1, 768]);  convert_element_type_34 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_31, view_31, primals_32);  primals_31 = None
        view_32 = torch.ops.aten.view.default(addmm_8, [1, 128, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_32, 768, 2);  view_32 = None
        getitem_24 = split_2[0]
        getitem_25 = split_2[1]
        getitem_26 = split_2[2];  split_2 = None
        view_33 = torch.ops.aten.view.default(getitem_25, [1, 128, -1, 64]);  getitem_25 = None
        permute_8 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        view_34 = torch.ops.aten.view.default(getitem_26, [1, 128, -1, 64]);  getitem_26 = None
        permute_9 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        view_35 = torch.ops.aten.view.default(getitem_24, [1, 128, -1, 64]);  getitem_24 = None
        permute_10 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        clone_11 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        clone_12 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        clone_13 = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_11, clone_12, clone_13, expand_1, True)
        getitem_27 = _scaled_dot_product_efficient_attention_2[0]
        getitem_28 = _scaled_dot_product_efficient_attention_2[1]
        getitem_29 = _scaled_dot_product_efficient_attention_2[2]
        getitem_30 = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        permute_11 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_36 = torch.ops.aten.view.default(permute_11, [1, 128, -1]);  permute_11 = None
        view_37 = torch.ops.aten.view.default(view_36, [-1, 768]);  view_36 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_33, view_37, primals_34);  primals_33 = view_37 = None
        view_38 = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
        add_20 = torch.ops.aten.add.Tensor(view_38, add_17);  view_38 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_41, [2], correction = 0, keepdim = True)
        getitem_31 = var_mean_5[0]
        getitem_32 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_41, getitem_32);  convert_element_type_41 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, primals_35);  mul_18 = None
        add_22 = torch.ops.aten.add.Tensor(mul_19, primals_36);  mul_19 = primals_36 = None
        convert_element_type_42 = torch.ops.prims.convert_element_type.default(add_22, torch.float16);  add_22 = None
        view_39 = torch.ops.aten.view.default(convert_element_type_42, [-1, 768]);  convert_element_type_42 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_37, view_39, primals_38);  primals_37 = None
        view_40 = torch.ops.aten.view.default(addmm_10, [1, 128, 3072])
        mul_20 = torch.ops.aten.mul.Tensor(view_40, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_40, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23 = torch.ops.aten.add.Tensor(view_40, mul_21);  view_40 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        view_41 = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_39, view_41, primals_40);  primals_39 = None
        view_42 = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
        add_25 = torch.ops.aten.add.Tensor(add_20, view_42);  view_42 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_49, [2], correction = 0, keepdim = True)
        getitem_33 = var_mean_6[0]
        getitem_34 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6 = torch.ops.aten.sub.Tensor(convert_element_type_49, getitem_34);  convert_element_type_49 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, primals_41);  mul_24 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, primals_42);  mul_25 = primals_42 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(add_27, torch.float16);  add_27 = None
        view_43 = torch.ops.aten.view.default(convert_element_type_50, [-1, 768]);  convert_element_type_50 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_43, view_43, primals_44);  primals_43 = None
        view_44 = torch.ops.aten.view.default(addmm_12, [1, 128, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_44, 768, 2);  view_44 = None
        getitem_35 = split_3[0]
        getitem_36 = split_3[1]
        getitem_37 = split_3[2];  split_3 = None
        view_45 = torch.ops.aten.view.default(getitem_36, [1, 128, -1, 64]);  getitem_36 = None
        permute_12 = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        view_46 = torch.ops.aten.view.default(getitem_37, [1, 128, -1, 64]);  getitem_37 = None
        permute_13 = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        view_47 = torch.ops.aten.view.default(getitem_35, [1, 128, -1, 64]);  getitem_35 = None
        permute_14 = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        clone_16 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        clone_17 = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        clone_18 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_16, clone_17, clone_18, expand_1, True)
        getitem_38 = _scaled_dot_product_efficient_attention_3[0]
        getitem_39 = _scaled_dot_product_efficient_attention_3[1]
        getitem_40 = _scaled_dot_product_efficient_attention_3[2]
        getitem_41 = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        permute_15 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_48 = torch.ops.aten.view.default(permute_15, [1, 128, -1]);  permute_15 = None
        view_49 = torch.ops.aten.view.default(view_48, [-1, 768]);  view_48 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_45, view_49, primals_46);  primals_45 = view_49 = None
        view_50 = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
        add_28 = torch.ops.aten.add.Tensor(view_50, add_25);  view_50 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_57, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_7[0]
        getitem_43 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7 = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_43);  convert_element_type_57 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, primals_47);  mul_26 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, primals_48);  mul_27 = primals_48 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None
        view_51 = torch.ops.aten.view.default(convert_element_type_58, [-1, 768]);  convert_element_type_58 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_49, view_51, primals_50);  primals_49 = None
        view_52 = torch.ops.aten.view.default(addmm_14, [1, 128, 3072])
        mul_28 = torch.ops.aten.mul.Tensor(view_52, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_52, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31 = torch.ops.aten.add.Tensor(view_52, mul_29);  view_52 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        view_53 = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_51, view_53, primals_52);  primals_51 = None
        view_54 = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
        add_33 = torch.ops.aten.add.Tensor(add_28, view_54);  view_54 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_65, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_8[0]
        getitem_45 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(convert_element_type_65, getitem_45);  convert_element_type_65 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, primals_53);  mul_32 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, primals_54);  mul_33 = primals_54 = None
        convert_element_type_66 = torch.ops.prims.convert_element_type.default(add_35, torch.float16);  add_35 = None
        view_55 = torch.ops.aten.view.default(convert_element_type_66, [-1, 768]);  convert_element_type_66 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_55, view_55, primals_56);  primals_55 = None
        view_56 = torch.ops.aten.view.default(addmm_16, [1, 128, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_56, 768, 2);  view_56 = None
        getitem_46 = split_4[0]
        getitem_47 = split_4[1]
        getitem_48 = split_4[2];  split_4 = None
        view_57 = torch.ops.aten.view.default(getitem_47, [1, 128, -1, 64]);  getitem_47 = None
        permute_16 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        view_58 = torch.ops.aten.view.default(getitem_48, [1, 128, -1, 64]);  getitem_48 = None
        permute_17 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        view_59 = torch.ops.aten.view.default(getitem_46, [1, 128, -1, 64]);  getitem_46 = None
        permute_18 = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        clone_21 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        clone_22 = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        clone_23 = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_21, clone_22, clone_23, expand_1, True)
        getitem_49 = _scaled_dot_product_efficient_attention_4[0]
        getitem_50 = _scaled_dot_product_efficient_attention_4[1]
        getitem_51 = _scaled_dot_product_efficient_attention_4[2]
        getitem_52 = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        permute_19 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_60 = torch.ops.aten.view.default(permute_19, [1, 128, -1]);  permute_19 = None
        view_61 = torch.ops.aten.view.default(view_60, [-1, 768]);  view_60 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_57, view_61, primals_58);  primals_57 = view_61 = None
        view_62 = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
        add_36 = torch.ops.aten.add.Tensor(view_62, add_33);  view_62 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_73, [2], correction = 0, keepdim = True)
        getitem_53 = var_mean_9[0]
        getitem_54 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9 = torch.ops.aten.sub.Tensor(convert_element_type_73, getitem_54);  convert_element_type_73 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, primals_59);  mul_34 = None
        add_38 = torch.ops.aten.add.Tensor(mul_35, primals_60);  mul_35 = primals_60 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(add_38, torch.float16);  add_38 = None
        view_63 = torch.ops.aten.view.default(convert_element_type_74, [-1, 768]);  convert_element_type_74 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_61, view_63, primals_62);  primals_61 = None
        view_64 = torch.ops.aten.view.default(addmm_18, [1, 128, 3072])
        mul_36 = torch.ops.aten.mul.Tensor(view_64, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_64, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39 = torch.ops.aten.add.Tensor(view_64, mul_37);  view_64 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        view_65 = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_63, view_65, primals_64);  primals_63 = None
        view_66 = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
        add_41 = torch.ops.aten.add.Tensor(add_36, view_66);  view_66 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_81, [2], correction = 0, keepdim = True)
        getitem_55 = var_mean_10[0]
        getitem_56 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10 = torch.ops.aten.sub.Tensor(convert_element_type_81, getitem_56);  convert_element_type_81 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, primals_65);  mul_40 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, primals_66);  mul_41 = primals_66 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None
        view_67 = torch.ops.aten.view.default(convert_element_type_82, [-1, 768]);  convert_element_type_82 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_67, view_67, primals_68);  primals_67 = None
        view_68 = torch.ops.aten.view.default(addmm_20, [1, 128, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_68, 768, 2);  view_68 = None
        getitem_57 = split_5[0]
        getitem_58 = split_5[1]
        getitem_59 = split_5[2];  split_5 = None
        view_69 = torch.ops.aten.view.default(getitem_58, [1, 128, -1, 64]);  getitem_58 = None
        permute_20 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        view_70 = torch.ops.aten.view.default(getitem_59, [1, 128, -1, 64]);  getitem_59 = None
        permute_21 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        view_71 = torch.ops.aten.view.default(getitem_57, [1, 128, -1, 64]);  getitem_57 = None
        permute_22 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_26 = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        clone_27 = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        clone_28 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_26, clone_27, clone_28, expand_1, True);  expand_1 = None
        getitem_60 = _scaled_dot_product_efficient_attention_5[0]
        getitem_61 = _scaled_dot_product_efficient_attention_5[1]
        getitem_62 = _scaled_dot_product_efficient_attention_5[2]
        getitem_63 = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        permute_23 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        view_72 = torch.ops.aten.view.default(permute_23, [1, 128, -1]);  permute_23 = None
        view_73 = torch.ops.aten.view.default(view_72, [-1, 768]);  view_72 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_69, view_73, primals_70);  primals_69 = view_73 = None
        view_74 = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
        add_44 = torch.ops.aten.add.Tensor(view_74, add_41);  view_74 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_89, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_11[0]
        getitem_65 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11 = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_65);  convert_element_type_89 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_71);  mul_42 = None
        add_46 = torch.ops.aten.add.Tensor(mul_43, primals_72);  mul_43 = primals_72 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(add_46, torch.float16);  add_46 = None
        view_75 = torch.ops.aten.view.default(convert_element_type_90, [-1, 768]);  convert_element_type_90 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_73, view_75, primals_74);  primals_73 = None
        view_76 = torch.ops.aten.view.default(addmm_22, [1, 128, 3072])
        mul_44 = torch.ops.aten.mul.Tensor(view_76, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_76, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47 = torch.ops.aten.add.Tensor(view_76, mul_45);  view_76 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        view_77 = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_75, view_77, primals_76);  primals_75 = None
        view_78 = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
        add_49 = torch.ops.aten.add.Tensor(add_44, view_78);  view_78 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_97, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_12[0]
        getitem_67 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12 = torch.ops.aten.sub.Tensor(convert_element_type_97, getitem_67);  convert_element_type_97 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, primals_77);  mul_48 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, primals_78);  mul_49 = primals_78 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None
        view_79 = torch.ops.aten.view.default(convert_element_type_98, [-1, 128, 768]);  convert_element_type_98 = None
        permute_24 = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_25 = torch.ops.aten.permute.default(view_77, [1, 0]);  view_77 = None
        permute_26 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_27 = torch.ops.aten.permute.default(view_75, [1, 0]);  view_75 = None
        permute_28 = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_34 = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_35 = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        permute_36 = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_37 = torch.ops.aten.permute.default(view_65, [1, 0]);  view_65 = None
        permute_38 = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_39 = torch.ops.aten.permute.default(view_63, [1, 0]);  view_63 = None
        permute_40 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_46 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_47 = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        permute_48 = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_49 = torch.ops.aten.permute.default(view_53, [1, 0]);  view_53 = None
        permute_50 = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_51 = torch.ops.aten.permute.default(view_51, [1, 0]);  view_51 = None
        permute_52 = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_58 = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_59 = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        permute_60 = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_61 = torch.ops.aten.permute.default(view_41, [1, 0]);  view_41 = None
        permute_62 = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_63 = torch.ops.aten.permute.default(view_39, [1, 0]);  view_39 = None
        permute_64 = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_70 = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_71 = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        permute_72 = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_73 = torch.ops.aten.permute.default(view_29, [1, 0]);  view_29 = None
        permute_74 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_75 = torch.ops.aten.permute.default(view_27, [1, 0]);  view_27 = None
        permute_76 = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_82 = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_83 = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        permute_84 = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_85 = torch.ops.aten.permute.default(view_17, [1, 0]);  view_17 = None
        permute_86 = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_87 = torch.ops.aten.permute.default(view_15, [1, 0]);  view_15 = None
        permute_88 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_94 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_95 = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        return (view_79, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, view, unsqueeze, add, getitem_1, rsqrt, clone_1, clone_2, clone_3, where, getitem_5, getitem_6, getitem_7, getitem_8, add_4, getitem_10, rsqrt_1, addmm_2, add_9, getitem_12, rsqrt_2, clone_6, clone_7, clone_8, getitem_16, getitem_17, getitem_18, getitem_19, add_12, getitem_21, rsqrt_3, addmm_6, add_17, getitem_23, rsqrt_4, clone_11, clone_12, clone_13, getitem_27, getitem_28, getitem_29, getitem_30, add_20, getitem_32, rsqrt_5, addmm_10, add_25, getitem_34, rsqrt_6, clone_16, clone_17, clone_18, getitem_38, getitem_39, getitem_40, getitem_41, add_28, getitem_43, rsqrt_7, addmm_14, add_33, getitem_45, rsqrt_8, clone_21, clone_22, clone_23, getitem_49, getitem_50, getitem_51, getitem_52, add_36, getitem_54, rsqrt_9, addmm_18, add_41, getitem_56, rsqrt_10, clone_26, clone_27, clone_28, getitem_60, getitem_61, getitem_62, getitem_63, add_44, getitem_65, rsqrt_11, addmm_22, add_49, getitem_67, rsqrt_12, permute_24, permute_25, permute_26, permute_27, permute_28, permute_34, permute_35, permute_36, permute_37, permute_38, permute_39, permute_40, permute_46, permute_47, permute_48, permute_49, permute_50, permute_51, permute_52, permute_58, permute_59, permute_60, permute_61, permute_62, permute_63, permute_64, permute_70, permute_71, permute_72, permute_73, permute_74, permute_75, permute_76, permute_82, permute_83, permute_84, permute_85, permute_86, permute_87, permute_88, permute_94, permute_95)
        
def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 77194752, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (50257, 768), dtype=torch.float16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1024, 768), dtype=torch.float16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf3, (1, 128), dtype=torch.int64, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (768,), dtype=torch.float16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (768,), dtype=torch.float16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (2304,), dtype=torch.float16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (768,), dtype=torch.float16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (768,), dtype=torch.float16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (768,), dtype=torch.float16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (3072,), dtype=torch.float16, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf14, (768,), dtype=torch.float16, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf15, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf16, (768,), dtype=torch.float16, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf17, (768,), dtype=torch.float16, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf18, (2304,), dtype=torch.float16, is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf19, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf20, (768,), dtype=torch.float16, is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf21, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf22, (768,), dtype=torch.float16, is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf23, (768,), dtype=torch.float16, is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf24, (3072,), dtype=torch.float16, is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf25, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf26, (768,), dtype=torch.float16, is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf27, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf28, (768,), dtype=torch.float16, is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf29, (768,), dtype=torch.float16, is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf30, (2304,), dtype=torch.float16, is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf31, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf32, (768,), dtype=torch.float16, is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf33, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf34, (768,), dtype=torch.float16, is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf35, (768,), dtype=torch.float16, is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf36, (3072,), dtype=torch.float16, is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf37, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf38, (768,), dtype=torch.float16, is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf39, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf40, (768,), dtype=torch.float16, is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf41, (768,), dtype=torch.float16, is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf42, (2304,), dtype=torch.float16, is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf43, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf44, (768,), dtype=torch.float16, is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf45, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf46, (768,), dtype=torch.float16, is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf47, (768,), dtype=torch.float16, is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf48, (3072,), dtype=torch.float16, is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf49, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf50, (768,), dtype=torch.float16, is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf51, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf52, (768,), dtype=torch.float16, is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf53, (768,), dtype=torch.float16, is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf54, (2304,), dtype=torch.float16, is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf55, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf56, (768,), dtype=torch.float16, is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf57, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf58, (768,), dtype=torch.float16, is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf59, (768,), dtype=torch.float16, is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf60, (3072,), dtype=torch.float16, is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf61, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf62, (768,), dtype=torch.float16, is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf63, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf64, (768,), dtype=torch.float16, is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf65, (768,), dtype=torch.float16, is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 4608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf66, (2304,), dtype=torch.float16, is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 3538944, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf67, (768, 2304), dtype=torch.float16, is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf68, (768,), dtype=torch.float16, is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf69, (768, 768), dtype=torch.float16, is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf70, (768,), dtype=torch.float16, is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf71, (768,), dtype=torch.float16, is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf72, (3072,), dtype=torch.float16, is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf73, (768, 3072), dtype=torch.float16, is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf74, (768,), dtype=torch.float16, is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf75, (3072, 768), dtype=torch.float16, is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf76, (768,), dtype=torch.float16, is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf77, (768,), dtype=torch.float16, is_leaf=True)  # primals_78
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)