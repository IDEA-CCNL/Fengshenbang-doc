# Wenzhong-GPT2-110M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLG任务，中文版的GPT2-Small。

Focused on handling NLG tasks, Chinese GPT2-Small.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言生成 NLG | 闻仲 Wenzhong | GPT2 |      110M      |     中文 Chinese     |

## 模型信息 Model Information

类似于Wenzhong2.0-GPT2-3.5B-chinese，我们实现了一个small版本的12层的Wenzhong-GPT2-110M，并且在悟道（300G版本）上面进行预训练。

Similar to Wenzhong2.0-GPT2-3.5B-chinese, we implement a small size Wenzhong-GPT2-110M with 12 layers, which is pre-trained on Wudao Corpus (300G version).

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Wenzhong-GPT2-110M](https://huggingface.co/IDEA-CCNL/Wenzhong-GPT2-110M)

### 加载模型 Loading Models

```python 
from transformers import GPT2Tokenizer,GPT2LMHeadModel
hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
```

### 使用示例 Usage Examples

```python
question = "北京是中国的"
inputs = tokenizer(question,return_tensors='pt')
generation_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=150,
                                # max_new_tokens=80,
                                do_sample=True,
                                top_p = 0.6,
                                # num_beams=5,
                                eos_token_id=50256,
                                pad_token_id=0,
                                num_return_sequences = 5)

for idx,sentence in enumerate(generation_output.sequences):
    print('next sentence %d:\n'%idx,
    tokenizer.decode(sentence).split('<|endoftext|>')[0])
    print('*'*40)
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
