# Taiyi-CLIP-Roberta-large-326M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

首个开源的中文CLIP模型，1.23亿图文对上进行预训练的文本端RoBERTa-large。

The first open source Chinese CLIP, pre-training on 123M image-text pairs, the text encoder: RoBERTa-large.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | CLIP (RoBERTa) |     326M    |    中文 Chinese     |

## 模型信息 Model Information

我们遵循CLIP的实验设置，以获得强大的视觉-语言表征。在训练中文版的CLIP时，我们使用[chinese-roberta-wwm-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)作为语言的编码器，并将[CLIP](https://github.com/openai/CLIP)中的**ViT-L-14**应用于视觉的编码器。为了快速且稳定地进行预训练，我们冻结了视觉编码器并且只微调语言编码器。此外，我们将[Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://zero.so.com/)数据集(23M)用作预训练的数据集。我们先在悟空数据集上预训练了10轮，然后接着在悟空数据集和zero数据集上预训练12轮。据我们所知，我们的Taiyi-CLIP是目前Huggingface社区中首个的开源中文CLIP。

We follow the experimental setup of CLIP to obtain powerful visual-language intelligence. To obtain the CLIP for Chinese, we employ [chinese-roberta-wwm-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) for the language encoder, and apply the **ViT-L-14** in [CLIP](https://github.com/openai/CLIP) for the vision encoder. We freeze the vision encoder and tune the language encoder to speed up and stabilize the pre-training process. Moreover, we apply [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) dataset (100M) and [Zero](https://zero.so.com/) dataset (23M) as the pre-training datasets. The model was first trained 10 epochs on wukong and then train another 12 epochs on wukong and zero. To the best of our knowledge, our TaiyiCLIP is currently the only open-sourced Chinese CLIP in the huggingface community.

### 下游效果 Performance

**Zero-Shot Classification**

|  model   | dataset  | Top1 | Top5 |
|  ----  | ----  | ---- | ---- |
| Taiyi-CLIP-Roberta-326M-Chinese  | ImageNet1k-CN | 53.05% | 79.55% |

**Zero-Shot Text-to-Image Retrieval**

|  model   | dataset  | Top1 | Top5 | Top10 |
|  ----  | ----  | ---- | ---- | ---- |
| Taiyi-CLIP-Roberta-326M-Chinese  | Flickr30k-CNA-test | 54.36% | 80.56%  | 87.90% |
| Taiyi-CLIP-Roberta-326M-Chinese  | COCO-CN-test | 51.47% | 81.00%  | 90.40% |
| Taiyi-CLIP-Roberta-326M-Chinese  | wukong50k | 61.18% | 90.46% | 95.74% |

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Taiyi-CLIP-Roberta-large-326M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese)

### 加载模型 Loading Models

```python3
from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np

query_texts = ["一只猫", "一只狗",'两只猫', '两只老虎','一只老虎']  # 这里是输入文本的，可以随意替换。
# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 这里可以换成任意图片的url
# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")
if image_data.mode != 'RGB':
    image = image.convert('RGB')

with torch.no_grad():
    image_features = clip_model.get_image_features(**image)
    text_features = text_encoder(text).logits
    # 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 计算余弦相似度 logit_scale是尺度系数
    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(np.around(probs, 3))

```

### 在下游任务微调 Finetuning

我们提供了CLIP在Flickr30k-CNA这个数据集上的finetune代码示例，另外我们也提供了召回率计算的代码，都集成在LightningModule里了。（案例是base版的，直接替换模型就可以用large版来finetune）

具体见：[地址](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/clip_finetune)

配置好相关环境后，执行
`sh finetune_flickr.sh`即可。

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
