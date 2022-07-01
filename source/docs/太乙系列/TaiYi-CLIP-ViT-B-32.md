# TaiYi-CLIP-ViT-B-32

Taiyi (太乙) 系列模型属于 Fengshenbang (封神榜) 大模型开源计划的一个分支。专注于利用多模态信息，使模型具有多模态语义理解能力。此模型为TaiYi-CLIP系列专注于CLIP在中文语料上的训练，作为中文图文多模态基础模型开源。

TaiYi-CLIP-ViT-B-32用的模型和[openai/CLIP](https://github.com/openai/CLIP) 的VIT-B-32一致，tokenizer也一致。（为了处理中文，这个tokenizer会把中文转为utf-8编码，然后再进行训练）。训练时，该模型加载原有的ViT-B-32的参数，然后在 [Noah-Wukong Dataset](https://wukong-dataset.github.io/wukong-dataset/) （大约一亿个图文对，我们实际只下载到了0.9亿）上训练20个epoch。

## 模型下载

[IDEA-CCNL/TaiYi-CLIP-ViT-B-32 · Hugging Face](https://huggingface.co/IDEA-CCNL/TaiYi-CLIP-ViT-B-32)

## 模型性能

| model               | dataset   | i2t-top1   | i2t-top5   | i2t-top10 | t2i-top1   | t2i-top5   | t2i-top10 |
| :------------------ | :-------- | :--------- | :--------- | :-------- | :--------- | :--------- | :-------- |
| Taiyi-CLIP-ViT-B/32 | wukong50k | **52.67%** | **77.92%** | 83.04%    | **51.22%** | **77.25%** | 82.85     |



## 快速开始

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
hg_model = CLIPModel.from_pretrained("IDEA-CCNL/TaiYi-CLIP-ViT-B-32")
hg_preprocess = CLIPProcessor.from_pretrained("IDEA-CCNL/TaiYi-CLIP-ViT-B-32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # load an image from Internet
image = Image.open(requests.get(url, stream=True).raw)

inputs = hg_preprocess(text=["一只猫", "一只狗",'两只猫', '两只老虎','一只老虎'], images=image, return_tensors="pt", padding=True)

outputs = hg_model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)

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