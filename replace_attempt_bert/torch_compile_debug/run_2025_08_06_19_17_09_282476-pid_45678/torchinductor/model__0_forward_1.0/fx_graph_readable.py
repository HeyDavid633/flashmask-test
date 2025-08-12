class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128]", primals_2: "f16[50257, 768]", primals_3: "f16[1024, 768]", primals_4: "i64[1, 128]", primals_5: "f16[768]", primals_6: "f16[768]", primals_7: "f16[2304]", primals_8: "f16[768, 2304]", primals_9: "f16[768]", primals_10: "f16[768, 768]", primals_11: "f16[768]", primals_12: "f16[768]", primals_13: "f16[3072]", primals_14: "f16[768, 3072]", primals_15: "f16[768]", primals_16: "f16[3072, 768]", primals_17: "f16[768]", primals_18: "f16[768]", primals_19: "f16[2304]", primals_20: "f16[768, 2304]", primals_21: "f16[768]", primals_22: "f16[768, 768]", primals_23: "f16[768]", primals_24: "f16[768]", primals_25: "f16[3072]", primals_26: "f16[768, 3072]", primals_27: "f16[768]", primals_28: "f16[3072, 768]", primals_29: "f16[768]", primals_30: "f16[768]", primals_31: "f16[2304]", primals_32: "f16[768, 2304]", primals_33: "f16[768]", primals_34: "f16[768, 768]", primals_35: "f16[768]", primals_36: "f16[768]", primals_37: "f16[3072]", primals_38: "f16[768, 3072]", primals_39: "f16[768]", primals_40: "f16[3072, 768]", primals_41: "f16[768]", primals_42: "f16[768]", primals_43: "f16[2304]", primals_44: "f16[768, 2304]", primals_45: "f16[768]", primals_46: "f16[768, 768]", primals_47: "f16[768]", primals_48: "f16[768]", primals_49: "f16[3072]", primals_50: "f16[768, 3072]", primals_51: "f16[768]", primals_52: "f16[3072, 768]", primals_53: "f16[768]", primals_54: "f16[768]", primals_55: "f16[2304]", primals_56: "f16[768, 2304]", primals_57: "f16[768]", primals_58: "f16[768, 768]", primals_59: "f16[768]", primals_60: "f16[768]", primals_61: "f16[3072]", primals_62: "f16[768, 3072]", primals_63: "f16[768]", primals_64: "f16[3072, 768]", primals_65: "f16[768]", primals_66: "f16[768]", primals_67: "f16[2304]", primals_68: "f16[768, 2304]", primals_69: "f16[768]", primals_70: "f16[768, 768]", primals_71: "f16[768]", primals_72: "f16[768]", primals_73: "f16[3072]", primals_74: "f16[768, 3072]", primals_75: "f16[768]", primals_76: "f16[3072, 768]", primals_77: "f16[768]", primals_78: "f16[768]"):
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:818 in forward, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: "i64[1, 128]" = torch.ops.aten.view.default(primals_1, [-1, 128]);  primals_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:857 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f16[1, 128, 768]" = torch.ops.aten.embedding.default(primals_2, view);  primals_2 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:861 in forward, code: cache_position = torch.arange(
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0)
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:867 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f16[1, 128, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:868 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:873 in forward, code: attention_mask = attention_mask.view(batch_size, -1)
        view_1: "i64[1, 128]" = torch.ops.aten.view.default(primals_4, [1, -1]);  primals_4 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/masking_utils.py:705 in _preprocess_mask_arguments, code: attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)
        convert_element_type: "b8[1, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.bool);  view_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/masking_utils.py:363 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_1: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0)
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/masking_utils.py:369 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /usr/local/lib/python3.12/dist-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:142 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_2: "i64[128, 1]" = torch.ops.aten.view.default(iota, [128, 1]);  iota = None
        le: "b8[128, 128]" = torch.ops.aten.le.Tensor(add_1, view_2);  view_2 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:142 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[128, 1]" = torch.ops.aten.full.default([128, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bitwise_and: "b8[128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        
         # File: /usr/local/lib/python3.12/dist-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        view_4: "i64[1, 1]" = torch.ops.aten.view.default(iota_2, [1, 1]);  iota_2 = None
        index: "b8[1, 128]" = torch.ops.aten.index.Tensor(convert_element_type, [view_4, add_1]);  convert_element_type = view_4 = add_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:142 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_5: "b8[1, 1, 128]" = torch.ops.aten.view.default(index, [1, 1, 128]);  index = None
        bitwise_and_1: "b8[1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, view_5);  bitwise_and = view_5 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/torch/_functorch/vmap.py:184 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        view_6: "b8[1, 1, 128, 128]" = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 128, 128]);  bitwise_and_1 = None
        expand: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(view_6, [1, 1, 128, 128]);  view_6 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_1: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = None
        mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        convert_element_type_2: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_2, [-1, 768]);  convert_element_type_2 = None
        addmm: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_7, view_7, primals_8);  primals_7 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_8: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm, [1, 128, 2304]);  addmm = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_8, 768, 2);  view_8 = None
        getitem_2: "f16[1, 128, 768]" = split[0]
        getitem_3: "f16[1, 128, 768]" = split[1]
        getitem_4: "f16[1, 128, 768]" = split[2];  split = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_9: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_3, [1, 128, -1, 64]);  getitem_3 = None
        permute: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_10: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_4, [1, 128, -1, 64]);  getitem_4 = None
        permute_1: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_11: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_2, [1, 128, -1, 64]);  getitem_2 = None
        permute_2: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_1: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_2: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_3: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[1, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  expand = full_default_2 = full_default_1 = None
        expand_1: "f16[1, 12, 128, 128]" = torch.ops.aten.expand.default(where, [1, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_1, clone_2, clone_3, expand_1, True)
        getitem_5: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_12: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_3, [1, 128, -1]);  permute_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f16[128, 768]" = torch.ops.aten.view.default(view_12, [-1, 768]);  view_12 = None
        addmm_1: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_9, view_13, primals_10);  primals_9 = view_13 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_14: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_4: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_14, add);  view_14 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_9: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_9, [2], correction = 0, keepdim = True)
        getitem_9: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_10: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_9, getitem_10);  convert_element_type_9 = None
        mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_11);  mul_2 = None
        add_6: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
        convert_element_type_10: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.float16);  add_6 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_15: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_10, [-1, 768]);  convert_element_type_10 = None
        addmm_2: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_13, view_15, primals_14);  primals_13 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_16: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
        pow_1: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_16, 3.0)
        mul_5: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_16, mul_5);  view_16 = mul_5 = None
        mul_6: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_17: "f16[128, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_15, view_17, primals_16);  primals_15 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_18: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_9: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_4, view_18);  view_18 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_17: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_12: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_12);  convert_element_type_17 = None
        mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_17);  mul_8 = None
        add_11: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_18);  mul_9 = primals_18 = None
        convert_element_type_18: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float16);  add_11 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_18, [-1, 768]);  convert_element_type_18 = None
        addmm_4: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_19, view_19, primals_20);  primals_19 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_20: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm_4, [1, 128, 2304]);  addmm_4 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_20, 768, 2);  view_20 = None
        getitem_13: "f16[1, 128, 768]" = split_1[0]
        getitem_14: "f16[1, 128, 768]" = split_1[1]
        getitem_15: "f16[1, 128, 768]" = split_1[2];  split_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_21: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_14, [1, 128, -1, 64]);  getitem_14 = None
        permute_4: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_22: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_15, [1, 128, -1, 64]);  getitem_15 = None
        permute_5: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_23: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_13, [1, 128, -1, 64]);  getitem_13 = None
        permute_6: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_6: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_7: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_8: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_6, clone_7, clone_8, expand_1, True)
        getitem_16: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_24: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_7, [1, 128, -1]);  permute_7 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f16[128, 768]" = torch.ops.aten.view.default(view_24, [-1, 768]);  view_24 = None
        addmm_5: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_21, view_25, primals_22);  primals_21 = view_25 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_26: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_12: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_26, add_9);  view_26 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_25: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_25, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_25, getitem_21);  convert_element_type_25 = None
        mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_11: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_23);  mul_10 = None
        add_14: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_24);  mul_11 = primals_24 = None
        convert_element_type_26: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_27: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_26, [-1, 768]);  convert_element_type_26 = None
        addmm_6: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_25, view_27, primals_26);  primals_25 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_28: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
        pow_2: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_28, 3.0)
        mul_13: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_28, mul_13);  view_28 = mul_13 = None
        mul_14: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_29: "f16[128, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_27, view_29, primals_28);  primals_27 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_30: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_17: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_12, view_30);  view_30 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_33: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_23: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_23);  convert_element_type_33 = None
        mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_29);  mul_16 = None
        add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_30);  mul_17 = primals_30 = None
        convert_element_type_34: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_34, [-1, 768]);  convert_element_type_34 = None
        addmm_8: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_31, view_31, primals_32);  primals_31 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_32: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm_8, [1, 128, 2304]);  addmm_8 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_32, 768, 2);  view_32 = None
        getitem_24: "f16[1, 128, 768]" = split_2[0]
        getitem_25: "f16[1, 128, 768]" = split_2[1]
        getitem_26: "f16[1, 128, 768]" = split_2[2];  split_2 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_33: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_25, [1, 128, -1, 64]);  getitem_25 = None
        permute_8: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_34: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_26, [1, 128, -1, 64]);  getitem_26 = None
        permute_9: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_35: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_24, [1, 128, -1, 64]);  getitem_24 = None
        permute_10: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_11: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_12: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_13: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_11, clone_12, clone_13, expand_1, True)
        getitem_27: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_36: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_11, [1, 128, -1]);  permute_11 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f16[128, 768]" = torch.ops.aten.view.default(view_36, [-1, 768]);  view_36 = None
        addmm_9: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_33, view_37, primals_34);  primals_33 = view_37 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_38: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_20: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_38, add_17);  view_38 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_41: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_41, [2], correction = 0, keepdim = True)
        getitem_31: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_32: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_41, getitem_32);  convert_element_type_41 = None
        mul_18: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_19: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_35);  mul_18 = None
        add_22: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_36);  mul_19 = primals_36 = None
        convert_element_type_42: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float16);  add_22 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_39: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_42, [-1, 768]);  convert_element_type_42 = None
        addmm_10: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_37, view_39, primals_38);  primals_37 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_40: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_40, 0.5)
        pow_3: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_40, 3.0)
        mul_21: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_40, mul_21);  view_40 = mul_21 = None
        mul_22: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_41: "f16[128, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_39, view_41, primals_40);  primals_39 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_42: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_25: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_20, view_42);  view_42 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_49: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_49, [2], correction = 0, keepdim = True)
        getitem_33: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_34: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_49, getitem_34);  convert_element_type_49 = None
        mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_25: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_41);  mul_24 = None
        add_27: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_42);  mul_25 = primals_42 = None
        convert_element_type_50: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.float16);  add_27 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_50, [-1, 768]);  convert_element_type_50 = None
        addmm_12: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_43, view_43, primals_44);  primals_43 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_44: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm_12, [1, 128, 2304]);  addmm_12 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_44, 768, 2);  view_44 = None
        getitem_35: "f16[1, 128, 768]" = split_3[0]
        getitem_36: "f16[1, 128, 768]" = split_3[1]
        getitem_37: "f16[1, 128, 768]" = split_3[2];  split_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_45: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_36, [1, 128, -1, 64]);  getitem_36 = None
        permute_12: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_46: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_37, [1, 128, -1, 64]);  getitem_37 = None
        permute_13: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_47: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_35, [1, 128, -1, 64]);  getitem_35 = None
        permute_14: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_16: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_17: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_18: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_16, clone_17, clone_18, expand_1, True)
        getitem_38: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_48: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_15, [1, 128, -1]);  permute_15 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f16[128, 768]" = torch.ops.aten.view.default(view_48, [-1, 768]);  view_48 = None
        addmm_13: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_45, view_49, primals_46);  primals_45 = view_49 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_50: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_28: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_50, add_25);  view_50 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_57: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_57, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_43: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_43);  convert_element_type_57 = None
        mul_26: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_27: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_47);  mul_26 = None
        add_30: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_48);  mul_27 = primals_48 = None
        convert_element_type_58: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_51: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_58, [-1, 768]);  convert_element_type_58 = None
        addmm_14: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_49, view_51, primals_50);  primals_49 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_52: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
        pow_4: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_52, 3.0)
        mul_29: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_52, mul_29);  view_52 = mul_29 = None
        mul_30: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_53: "f16[128, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_51, view_53, primals_52);  primals_51 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_54: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_33: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_28, view_54);  view_54 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_65: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_65, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_45: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_65, getitem_45);  convert_element_type_65 = None
        mul_32: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_33: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_53);  mul_32 = None
        add_35: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_54);  mul_33 = primals_54 = None
        convert_element_type_66: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.float16);  add_35 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_66, [-1, 768]);  convert_element_type_66 = None
        addmm_16: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_55, view_55, primals_56);  primals_55 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_56: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm_16, [1, 128, 2304]);  addmm_16 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_56, 768, 2);  view_56 = None
        getitem_46: "f16[1, 128, 768]" = split_4[0]
        getitem_47: "f16[1, 128, 768]" = split_4[1]
        getitem_48: "f16[1, 128, 768]" = split_4[2];  split_4 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_57: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_47, [1, 128, -1, 64]);  getitem_47 = None
        permute_16: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_58: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_48, [1, 128, -1, 64]);  getitem_48 = None
        permute_17: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_59: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_46, [1, 128, -1, 64]);  getitem_46 = None
        permute_18: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_21: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_22: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_23: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_21, clone_22, clone_23, expand_1, True)
        getitem_49: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_60: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_19, [1, 128, -1]);  permute_19 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f16[128, 768]" = torch.ops.aten.view.default(view_60, [-1, 768]);  view_60 = None
        addmm_17: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_57, view_61, primals_58);  primals_57 = view_61 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_62: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_36: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_62, add_33);  view_62 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_73: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_73, [2], correction = 0, keepdim = True)
        getitem_53: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_54: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_73, getitem_54);  convert_element_type_73 = None
        mul_34: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_59);  mul_34 = None
        add_38: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_60);  mul_35 = primals_60 = None
        convert_element_type_74: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_38, torch.float16);  add_38 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_63: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_74, [-1, 768]);  convert_element_type_74 = None
        addmm_18: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_61, view_63, primals_62);  primals_61 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_64: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_64, 0.5)
        pow_5: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_64, 3.0)
        mul_37: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_64, mul_37);  view_64 = mul_37 = None
        mul_38: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_65: "f16[128, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_63, view_65, primals_64);  primals_63 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_66: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_41: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_36, view_66);  view_66 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:410 in forward, code: hidden_states = self.ln_1(hidden_states)
        convert_element_type_81: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_81, [2], correction = 0, keepdim = True)
        getitem_55: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_56: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_81, getitem_56);  convert_element_type_81 = None
        mul_40: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_41: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_65);  mul_40 = None
        add_43: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_66);  mul_41 = primals_66 = None
        convert_element_type_82: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_82, [-1, 768]);  convert_element_type_82 = None
        addmm_20: "f16[128, 2304]" = torch.ops.aten.addmm.default(primals_67, view_67, primals_68);  primals_67 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_68: "f16[1, 128, 2304]" = torch.ops.aten.view.default(addmm_20, [1, 128, 2304]);  addmm_20 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:312 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_68, 768, 2);  view_68 = None
        getitem_57: "f16[1, 128, 768]" = split_5[0]
        getitem_58: "f16[1, 128, 768]" = split_5[1]
        getitem_59: "f16[1, 128, 768]" = split_5[2];  split_5 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_69: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_58, [1, 128, -1, 64]);  getitem_58 = None
        permute_20: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_70: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_59, [1, 128, -1, 64]);  getitem_59 = None
        permute_21: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:318 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_71: "f16[1, 128, 12, 64]" = torch.ops.aten.view.default(getitem_57, [1, 128, -1, 64]);  getitem_57 = None
        permute_22: "f16[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:72 in sdpa_attention_forward, code: query = query.contiguous()
        clone_26: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: key = key.contiguous()
        clone_27: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:74 in sdpa_attention_forward, code: value = value.contiguous()
        clone_28: "f16[1, 12, 128, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:89 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_26, clone_27, clone_28, expand_1, True);  expand_1 = None
        getitem_60: "f16[1, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[1, 12, 128]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/integrations/sdpa_attention.py:99 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f16[1, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:356 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_72: "f16[1, 128, 768]" = torch.ops.aten.view.default(permute_23, [1, 128, -1]);  permute_23 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f16[128, 768]" = torch.ops.aten.view.default(view_72, [-1, 768]);  view_72 = None
        addmm_21: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_69, view_73, primals_70);  primals_69 = view_73 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_74: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:422 in forward, code: hidden_states = attn_output + residual
        add_44: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(view_74, add_41);  view_74 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:446 in forward, code: hidden_states = self.ln_2(hidden_states)
        convert_element_type_89: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_89, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_65: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_65);  convert_element_type_89 = None
        mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_71);  mul_42 = None
        add_46: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_72);  mul_43 = primals_72 = None
        convert_element_type_90: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_46, torch.float16);  add_46 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_75: "f16[128, 768]" = torch.ops.aten.view.default(convert_element_type_90, [-1, 768]);  convert_element_type_90 = None
        addmm_22: "f16[128, 3072]" = torch.ops.aten.addmm.default(primals_73, view_75, primals_74);  primals_73 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_76: "f16[1, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 128, 3072])
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/activations.py:47 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
        pow_6: "f16[1, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_76, 3.0)
        mul_45: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(view_76, mul_45);  view_76 = mul_45 = None
        mul_46: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f16[1, 128, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f16[1, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f16[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_77: "f16[128, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f16[128, 768]" = torch.ops.aten.addmm.default(primals_75, view_77, primals_76);  primals_75 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:117 in forward, code: x = x.view(size_out)
        view_78: "f16[1, 128, 768]" = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_49: "f16[1, 128, 768]" = torch.ops.aten.add.Tensor(add_44, view_78);  view_78 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:953 in forward, code: hidden_states = self.ln_f(hidden_states)
        convert_element_type_97: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_97, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_67: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_97, getitem_67);  convert_element_type_97 = None
        mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_49: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_77);  mul_48 = None
        add_51: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_78);  mul_49 = primals_78 = None
        convert_element_type_98: "f16[1, 128, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/models/gpt2/modeling_gpt2.py:955 in forward, code: hidden_states = hidden_states.view(output_shape)
        view_79: "f16[1, 128, 768]" = torch.ops.aten.view.default(convert_element_type_98, [-1, 128, 768]);  convert_element_type_98 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_24: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_25: "f16[3072, 128]" = torch.ops.aten.permute.default(view_77, [1, 0]);  view_77 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_26: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_27: "f16[768, 128]" = torch.ops.aten.permute.default(view_75, [1, 0]);  view_75 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_28: "f16[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_34: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_35: "f16[768, 128]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_36: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_37: "f16[3072, 128]" = torch.ops.aten.permute.default(view_65, [1, 0]);  view_65 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_38: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_39: "f16[768, 128]" = torch.ops.aten.permute.default(view_63, [1, 0]);  view_63 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_40: "f16[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_46: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_47: "f16[768, 128]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_48: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_49: "f16[3072, 128]" = torch.ops.aten.permute.default(view_53, [1, 0]);  view_53 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_50: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_51: "f16[768, 128]" = torch.ops.aten.permute.default(view_51, [1, 0]);  view_51 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_52: "f16[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_58: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_59: "f16[768, 128]" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_60: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_61: "f16[3072, 128]" = torch.ops.aten.permute.default(view_41, [1, 0]);  view_41 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_62: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_63: "f16[768, 128]" = torch.ops.aten.permute.default(view_39, [1, 0]);  view_39 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_64: "f16[768, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_70: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_71: "f16[768, 128]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_72: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_73: "f16[3072, 128]" = torch.ops.aten.permute.default(view_29, [1, 0]);  view_29 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_74: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_75: "f16[768, 128]" = torch.ops.aten.permute.default(view_27, [1, 0]);  view_27 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_76: "f16[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_82: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_83: "f16[768, 128]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_84: "f16[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_85: "f16[3072, 128]" = torch.ops.aten.permute.default(view_17, [1, 0]);  view_17 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_86: "f16[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_87: "f16[768, 128]" = torch.ops.aten.permute.default(view_15, [1, 0]);  view_15 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_88: "f16[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        
         # File: /usr/local/lib/python3.12/dist-packages/transformers/pytorch_utils.py:116 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_94: "f16[2304, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_95: "f16[768, 128]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        return (view_79, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, view, unsqueeze, add, getitem_1, rsqrt, clone_1, clone_2, clone_3, where, getitem_5, getitem_6, getitem_7, getitem_8, add_4, getitem_10, rsqrt_1, addmm_2, add_9, getitem_12, rsqrt_2, clone_6, clone_7, clone_8, getitem_16, getitem_17, getitem_18, getitem_19, add_12, getitem_21, rsqrt_3, addmm_6, add_17, getitem_23, rsqrt_4, clone_11, clone_12, clone_13, getitem_27, getitem_28, getitem_29, getitem_30, add_20, getitem_32, rsqrt_5, addmm_10, add_25, getitem_34, rsqrt_6, clone_16, clone_17, clone_18, getitem_38, getitem_39, getitem_40, getitem_41, add_28, getitem_43, rsqrt_7, addmm_14, add_33, getitem_45, rsqrt_8, clone_21, clone_22, clone_23, getitem_49, getitem_50, getitem_51, getitem_52, add_36, getitem_54, rsqrt_9, addmm_18, add_41, getitem_56, rsqrt_10, clone_26, clone_27, clone_28, getitem_60, getitem_61, getitem_62, getitem_63, add_44, getitem_65, rsqrt_11, addmm_22, add_49, getitem_67, rsqrt_12, permute_24, permute_25, permute_26, permute_27, permute_28, permute_34, permute_35, permute_36, permute_37, permute_38, permute_39, permute_40, permute_46, permute_47, permute_48, permute_49, permute_50, permute_51, permute_52, permute_58, permute_59, permute_60, permute_61, permute_62, permute_63, permute_64, permute_70, permute_71, permute_72, permute_73, permute_74, permute_75, permute_76, permute_82, permute_83, permute_84, permute_85, permute_86, permute_87, permute_88, permute_94, permute_95)
        