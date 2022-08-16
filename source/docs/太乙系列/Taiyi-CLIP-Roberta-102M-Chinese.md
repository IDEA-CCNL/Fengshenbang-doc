# Taiyi-CLIP-Roberta-102M-Chinese

Taiyi (太乙) 系列模型属于 Fengshenbang (封神榜) 大模型开源计划的一个分支。专注于利用多模态信息，使模型具有多模态语义理解能力。此模型为TaiYi-CLIP系列专注于CLIP在中文语料上的训练，作为中文图文多模态基础模型开源。

Taiyi-CLIP-Roberta-102M-Chinese使用[openai/CLIP](https://github.com/openai/CLIP) VIT-B-32作为image encoder，使用[chinese-roberta-wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext)作为text encoder，在 [Noah-Wukong Dataset](https://wukong-dataset.github.io/wukong-dataset/) （大约一亿个图文对，我们实际只下载到了0.9亿）上训练20个epoch。模型使用8张A100训练10天。

## 模型下载

[IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese)

## 模型性能
### Zero-Shot Classification
|  model   | dataset  | Top1 | Top5 |
|  ----  | ----  | ---- | ---- |
| Taiyi-CLIP-Roberta-102M-Chinese  | ImageNet1k-CN | 41.00% | 69.19% |

### Zero-Shot Text-to-Image Retrieval

|  model   | dataset  | Top1 | Top5 | Top10 |
|  ----  | ----  | ---- | ---- | ---- |
| Taiyi-CLIP-Roberta-102M-Chinese  | Flickr30k-CNA-test | 44.06% | 71.42%  | 80.84% |
| Taiyi-CLIP-Roberta-102M-Chinese  | COCO-CN-test | 46.24% | 78.06%  | 88.88% |
| Taiyi-CLIP-Roberta-102M-Chinese  | wukong50k | 48.67% | 81.77% | 90.09% |



## 使用示例
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
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 这里可以换成任意图片的url
# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")

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

## 下游Finetune
我们提供了CLIP在Flickr30k-CNA这个数据集上的finetune代码示例，另外我们也提供了召回率计算的代码，都集成在LightningModule里了。

具体见:https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/clip_finetune

配置好相关环境后，执行
`sh finetune_flickr.sh`即可。


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