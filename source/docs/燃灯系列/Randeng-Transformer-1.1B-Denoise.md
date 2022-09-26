# Randeng-Transformer-1.1B-Denoise

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

以去噪任务为微调目标的中文Transformer-XL。

Chinese Transformer-XL with a denoising task as the fine-tuning objective.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | Transformer |      1.1B      |     中文-去噪 Chinese-Denoise    |

## 模型信息 Model Information

我们先使用Transformer-XL的模型结构在悟道语料库(180G版本)上进行预训练，然后在我们自主构建的去噪数据集上进行微调。其中，去噪任务是从包括 **随机插入/交换/删除/替换/句子重排** 的具有噪声的输入中重建一个流畅和干净的文本。

We first pre-trained Transformer-XL on the Wudo corpus (180G version), and then fine-tuned it on a denoised dataset (developed by us). The denoise task is to reconstruct a fluent and clean text from a noisy input which includes **random insertion/swap/deletion/replacement/sentence reordering**.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Randeng-Transformer-1.1B-Denoise](Randeng-Transformer-1.1B-Denoise)

### 加载模型 Loading Models

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

```python 
from fengshen.models.transfo_xl_denoise.tokenization_transfo_xl_denoise import TransfoXLDenoiseTokenizer
from fengshen.models.transfo_xl_denoise.modeling_transfo_xl_denoise import TransfoXLDenoiseModel

tokenizer = TransfoXLDenoiseTokenizer.from_pretrained('IDEA-CCNL/Randeng-Transformer-1.1B-Denoise')
model = TransfoXLDenoiseModel.from_pretrained('IDEA-CCNL/Randeng-Transformer-1.1B-Denoise')
```

### 使用示例 Usage Examples

```python 
from fengshen.models.transfo_xl_denoise.generate import denoise_generate
input_text = "凡是有成就的人, 都很严肃地对待生命自己的"
res = denoise_generate(model, tokenizer,  input_text)
print(res)
# "有成就的人都很严肃地对待自己的生命。"
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
