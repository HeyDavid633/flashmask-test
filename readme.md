# fusion-SC25

## 对比对象
+ 可立即部署 PyTorch Naive、Torch.compile、TVM
+ 待部署 FasterTransformer、TVM、MCFuser、Ansor、Welder

## 端到端测试模型
|        Model             |  L   |  H   |  A   | W |
| :-------------------------: | :--: | :--: | :--: | :-------: |
| BERT-small  35M 【Encoder】 |  6   | 512  |  8   |    64     |
| BERT-base 110M 【Encoder】  |  12  | 768  |  12  |    64     |
| BERT-large 340M【Encoder】  |  24  | 1024 |  16  |    64     |
|    GPT（2/3）【Decoder】    |  12  | 768  |  12  |    64     |
|    T5【encoder+decoder】   |  12  | 768  |  12  |    64     |
|    llama3.1-8B【Encoder】  |  32  | 4096 |  32  |    64     |
|    llama3.1-70B【Decoder】 |  32  | 8192 |  32  |    64     |


## 算子批量测试尺寸
> 所有测试，在横轴上尽量保持为6个比较好 (6个seqlen)
> 测试精度完全是 FP16；除非特别说明，其中 L=12， H=768， A=12
> 关于Attention算子的尺寸设置参考 FlashAttention2（最大到16K）、Raptor-T（1k - 4k）、ByteTrans（64-1k）
> 关于Triton融合算子的取值，参考的MCFuser和Chimera

+ Attention算子
  + 固定 hidden_dim = 768; head_num = 12; head_size = 64
  + seq_len = 128, 256, 512, 1024, 2048, 4096 
  + Batch_size = 1, 8, 16 
  + 如果考虑的是TFLOPS为单位的话：Batch_size * seq_len = 8k 
+ Tirton ｜ TileLang 算子
  + Batch GEMM chain 谁快用谁
    + (B, S, H) @ (B, H, 4H)=(B, S, 4H);   (B, S, 4H)@(B, 4H, H)=(B, S, H);
    + 如果做批量测试（B, S, H）=  (1/8/16, 512/1024/2048, 64/128/256)
  + GEMM + bias + act：
    + (B, S, H)  同上
  + GEMM + layerNorm：
    + (B, S, H) @ (B, H, H) = (B, S, H)  ---layerNorm---> (B, S, H) 
    + 经典取值(B, S, H)=(8, 512, 64)、**(8, 1024, 768)**;
