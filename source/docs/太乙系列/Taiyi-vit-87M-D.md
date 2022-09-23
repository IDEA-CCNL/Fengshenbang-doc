# Taiyi-vit-87M-D

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

COCO和VG上特殊预训练的，英文版的MAP（名称暂定）的视觉端ViT-base。

Special pre-training on COCO and VG, the visual encoder for MAP (temporary) in English, ViT-base.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | 待定 TBD |      89M      |     特殊预训练方法 D     |

## 模型信息 Model Information

基于clip-vit-base (patch 16, resolution 224x224)，我们使用特殊的训练任务引入一些多模态信息。"D"表示这是一种新的预训练方法。对于特殊的多模态表征，在论文中我们设计了集中不同的训练目标。预训练数据集为MSCOCO和VG。我们的代码和预训练任务的细节将在论文接受后公开。

Based on pre-trained clip-vit-base (patch 16, resolution 224x224), we apply some multimodal information with special pre-training tasks. "D" implies a special training method. For special multimodal representations, we design several special training objectives in our paper. The pre-training datasets are MSCOCO and VG. Our code and details of pre-training tasks will be made publicly available upon paper acceptance.

### 下游任务 Performance

|                                      | CIFAR10 | ImageNet1k |
|--------------------------------------|:-------:|:----------:|
| clip-vit-base-patch16-224 (official) |   96.2  |    80.2    |
| Taiyi-vit-87M-D (local)              |   98.7  |    82.4    |

The local test settings are:

learning rate=2e-5, 
batch size=128, 
num train epochs=5, 
weight decay=0.01

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Taiyi-vit-87M-D](https://huggingface.co/IDEA-CCNL/Taiyi-vit-87M-D)

### 加载模型 Loading Models

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('IDEA-CCNL/Taiyi-vit-87M-D')
model = ViTForImageClassification.from_pretrained('IDEA-CCNL/Taiyi-vit-87M-D')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: Egyptian cat
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
