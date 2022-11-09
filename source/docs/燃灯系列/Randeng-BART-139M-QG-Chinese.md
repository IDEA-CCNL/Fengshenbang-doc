# Randeng-BART-139M-QG-Chinese

- Github: [Fengshenbang-LM/finetune_bart_qg](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/finetune_bart_qg)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理问题生成任务的中文版 BART-base 模型。

Good at solving question generation tasks Bart-base Model (Chinese version).

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | BART |      139M      |    问题生成任务-中文 QuestionGeneration-Chinese    |


## 模型信息 Model Information

基于[IDEA-CCNL/Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M)，我们在 [ChineseSQuAD](https://github.com/pluto-junzeng/ChineseSquad) 数据集上微调了问题生成任务版本。该数据集翻译了部分SQuAD数据集，包含约 67k 有答案的训练样本。

Based on [IDEA-CCNL/Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M), we fine-tuned a question generation version on [ChineseSQuAD](https://github.com/pluto-junzeng/ChineseSquad) datasets. The dataset is translated from SQuAD 2.0, with around 67k samples with answer.

### 下游效果 Performance
| Dataset          |  Size  | BLEU-4 | METEOR | ROUGE-L| 
| ------------ | -----  | -------- |--------- | ---------- |
|   ChineseSQuAD               |  139M   |  22.17 |   40.38  |   38.17   | 

## 使用 Usage

```python
from transformers import AutoTokenizer, BartForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Randeng-BART-139M-QG-Chinese",additional_special_tokens=["<ans>"])
model = BartForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-BART-139M-QG-Chinese")
context = "知识：1939年9月1日德国入侵波兰后，第二次世界大战开始，华沙一直被保卫到9月27日。波兰中部，包括华沙，都在德国纳粹殖民地政府总政府的统治下。所有的高等教育机构都立即关闭，华沙的犹太人口——几十万，约占城市的 <ans> ——全部涌入华沙的贫民区。回答：30%"
inputs = tokenizer.encode_plus(
            context,
            max_length=448,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
out = model.generate(                
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,
        num_beams=5,
        max_length=64,
        top_p = 0.9,
    )
pred = tokenizer.batch_decode(out,clean_up_tokenization_spaces=True, skip_special_tokens=True)[0]
print(pred)
# 问题:华沙的犹太人口占城市的百分之多少?
```


## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2210.08590)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2210.08590):

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
