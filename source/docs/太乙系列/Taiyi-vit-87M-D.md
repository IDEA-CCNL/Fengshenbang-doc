# Taiyi-vit-87M-D

Taiyi (太乙) 系列模型属于 Fengshenbang (封神榜) 大模型开源计划的一个分支。专注于利用多模态信息，使模型具有多模态语义理解能力。

此模型与 Taiyi-Roberta-124M-D，一同构成多模态双流模型的两个特征提取器。

我们使用了特殊的多模态预训练策略增强了不仅仅是多模态下游任务的表现性能，也增强了单模态下游任务的能力。预训练策略以及代码将会在论文被接收后一同开源。

其中，所用的vit基于预训练的clip-vit-base (patch 16, resolution 224x224)。"D"为特殊的预训练策略的名称缩写。此模型用于图片的特征抽取。

## 模型下载

[Huggingface Taiyi-vit-87M-D](https://huggingface.co/IDEA-CCNL/Taiyi-vit-87M-D)

## CV下游任务

|                                      | CIFAR10 | ImageNet1k |
|--------------------------------------|:-------:|:----------:|
| clip-vit-base-patch16-224 (official) |   96.2  |    80.2    |
| Taiyi-vit-87M-D (local)              |   98.7  |    82.4    |

local (本地)的设置:

learning_rate=2e-5, 
batch_size=128, 
num_train_epochs=5, 
weight_decay=0.01

## 快速开始

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

## Citation
If you find the resource is useful, please cite the following website in your paper.
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2022},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
