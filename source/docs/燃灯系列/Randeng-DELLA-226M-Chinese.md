# Randeng-DELLA-226M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

在悟道数据集上进行通用预训练的Deep VAE模型。其中编码器和解码器都是GPT-2架构。可以用于下游的句子重写，语义转换，性质控制等任务。

A deep VAE model pretrained on Wudao dataset. Both encoder and decoder are based on GPT-2 architecture. Such model is particularly suitable for paraphrasing, semantic updating and fine-grained attributes control.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言生成 NLG | 燃灯 Randeng | DELLA |     226M      |    变分自编码器-中文 VAE-Chinese    |


## 模型信息 Model Information

参考论文 Reference Paper：[Fuse It More Deeply! A Variational Transformer with Layer-Wise Latent Variable Inference for Text Generation](https://arxiv.org/abs/2207.06130)

本模型使用了Della论文里的循环潜在向量架构，但对于解码器生成并未采用原论文的low-rank-tensor-product来进行信息融合，而是使用了简单的线性变换后逐位逐词添加的方式。该方式对于开放域数据集的预训练稳定性有较大正向作用。

Note that although we adopted the layer-wise recurrent latent variables structure as the paper, we did not use the low-rank-tensor-product to fuse the latent vectors to the decoder hidden states. Instead we applied a simple linear transformation on the latent vectors and then add them to the hidden states independently. 


## 使用 Usage

```python
# Checkout the latest Fengshenbang-LM directory and run following script under Fengshenbang-LM root directory 
import torch
from torch.nn.utils.rnn import pad_sequence
from fengshen.models.deepVAE.deep_vae import Della
from transformers.models.bert.tokenization_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Randeng-DELLA-226M-Chinese")
vae_model = Della.from_pretrained("IDEA-CCNL/Randeng-DELLA-226M-Chinese")
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
tokenizer.add_special_tokens(special_tokens_dict)
sentence =  "本模型是在通用数据集下预训练的VAE模型，如要获得最佳效果请在特定领域微调后使用。"
tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
decoder_target = [tokenizer.bos_token_id] + tokenized_text + [tokenizer.eos_token_id]
inputs = []
inputs.append(torch.tensor(decoder_target, dtype=torch.long))
inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
max_length = 256
top_p = 0.5
top_k = 0
temperature = .7
repetition_penalty = 1.0
sample = False
device = 0
model = vae_model.eval()
model = model.to(device)
outputs = model.model.inference(inputs.to(device), top_p=top_p, top_k=top_k, max_length=max_length, sample=sample,
    temperature=temperature, repetition_penalty=repetition_penalty)
for gen_sent, orig_sent in zip(outputs, inputs):
    print('orig_sent:', tokenizer.decode(orig_sent).replace(' ', ''))
    print('gen_sent:', tokenizer.decode(gen_sent).replace(' ', ''))
    print("-"*20)
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen and Ruyi Gan and Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```