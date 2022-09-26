# YuyuanQA-GPT2-3.5B

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理医疗问答任务，医疗的领域模型，英文版的GPT2。

Good at handling medical question answering tasks, a medical domain model, GPT2 in English.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 领域 Domain | 余元 Yuyuan | GPT2 |      3.5B      |     问答 QA    |

## 模型信息 Model Information

问答在自然语言处理领域中反映AI系统的知识水平的重要任务。为了可以在医疗领域中使用强大的问答能力的语言模型，我们基于Yuyuan-GPT2-3.5B，对其使用了10K条医疗的问答对进行微调。我们希望探索一种简单、有效的方式直接实现问答系统而不需要额外的设计，即利用大模型强大的记忆力和理解能力。 

Question answering (QA) is an important task in the Natural Language Processing to present the knowledge level of AI systems. To provide a language model with powerful QA capability in the medical domain, we fine-tuned Yuyuan-GPT2-3.5B on 10K medical Q&A pairs. 

### 模型 Model

finetune的模型是yuyuan模型，余元模型是GPT2的结构，在预训练阶段主要是用PubMed医疗相关的数据集进行的预训练。是一个医疗领域的大模型。模型共有35亿参数，主要参数如下表所示：

|    配置     | 参数  |
| :---------: | :---: |
|   nlayers   |  30   |
|  nheaders   |  32   |
| hidden-size | 3072  |
| seq-length  | 1024  |

预训练的数据，主要医疗相关的论文、杂志期刊等，以英文语料为主。

### 数据 Data

用于finetune的语料是清洗于[MedQuAD](https://github.com/abachaa/MedQuAD)数据集，清洗完成后是下面的格式：
```text
......
{'question':'.........','answer':'........'}
{'question':'.........','answer':'........'}
......
```

### 框架 Framework

finetune的框架是IDEA研究院CCNL小组整合各大框架的优点开源的[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)，具体代码详见：Fengshenbang-LM/fengshen/examples/wenzhong_qa/finetune_medicalQA.py 和 Fengshenbang-LM/fengshen/data/task_dataloader/medicalQADataset.py。

### 训练参数 Parameter

训练参数，我们采用了deepspeed相关的配置，用2个集群的节点共16张A100，在很短的时间内完成了finetune。具体参数配置可以参考：Fengshenbang-LM/fengshen/examples/wenzhong_qa/finetune_GPT2_medicalQA.sh

### 效果对比 Results

finetune后的模型，用100对问答对，基于BLEU分与之前用Megatron框架训练的模型进行了简单的对比，效果比较接近。

unsmoth method:

| 框架     | 1-gram             | 2-gram             | 3-gram             | 4-gram              |
| -------- | ------------------ | ------------------ | ------------------ | ------------------- |
| Fengshen | 0.5241376169070796 | 0.5215762466122144 | 0.4894353584800885 | 0.44840139357073466 |
| Megatron | 0.5321340489166898 | 0.5110257474778213 | 0.4703745962926368 | 0.4310875933354554  |

smoth method:

| 框架     | 1-gram            | 2-gram             | 3-gram             | 4-gram             |
| -------- | ----------------- | ------------------ | ------------------ | ------------------ |
| Fengshen | 0.717829796617609 | 0.6516910802858905 | 0.5859726677095979 | 0.525510691686505  |
| Megatron | 0.776190980974117 | 0.6749801211321476 | 0.5897846253142169 | 0.5230773076722481 |

### 下游任务 Performance

我们测试了该模型在未见过的100条QA对上的表现：

We tested the model on 100 unseen QA pairs:

| gram | 1-gram | 2-gram | 3-gram | 4-gram |
| :----: | :----: |:----: | :----: | :----: |
| blue score  | 0.357727 | 0.2713 | 0.22304 | 0.19099 |

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：YuyuanQA-GPT2-3.5B](https://huggingface.co/IDEA-CCNL/YuyuanQA-GPT2-3.5B)

### 加载模型 Loading Models

```python 
from transformers import GPT2Tokenizer,GPT2LMHeadModel

hf_model_path = 'YuyuanQA-GPT2-3.5B'

tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
```

### 使用示例 Usage Examples

```python
fquestion = "What should gout patients pay attention to in diet?"
inputs = tokenizer(f'Question:{question} answer:',return_tensors='pt')

generation_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=150,
                                # max_new_tokens=80,
                                do_sample=True,
                                top_p = 0.6,
                                eos_token_id=50256,
                                pad_token_id=0,
                                num_return_sequences = 5)

for idx,sentence in enumerate(generation_output.sequences):
    print('next sentence %d:\n'%idx,
          tokenizer.decode(sentence).split('<|endoftext|>')[0])
    print('*'*40)

```

### 回答问题 Answering the Questions

支持直接用Haggingface或者pytorch-lightning框架调用。由于在finetune的时候，加入了prompt，在问答的时候，输入应该是："`Question:your question about medical? answer:`",接着模型就回以续写的方式回答你的问题。用huggingface的调用代码可以参考下面的代码：

```python 
from transformers import GPT2Tokenizer,GPT2LMHeadModel
model_path = 'pretrained_model_hf/yuyuanQA-v1' # input your own model file path
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = model.cuda(6) # move your model to the GPU
model.eval() # just do predict

def answering(question):
# question = "What should gout patients pay attention to in diet?"
    inputs = tokenizer(f'Question:{question} answer:',return_tensors='pt').input_ids.to(model.device)
    
    generation_output = model.generate(input_ids = inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=150,
                                # max_new_tokens=80,
                                do_sample=True,
                                top_p = 0.9,
                                eos_token_id=50256,
                                pad_token_id=0,
                                num_return_sequences = 5)
    answers = []
    for idx,sentence in enumerate(generation_output.sequences):
        next_sentence = tokenizer.decode(sentence).split('<|endoftext|>')[0]
        answer = next_sentence.split(sep='answer:',maxsplit=1)[1]
        answers.append(answer)
    return answers
answering('your question?')
```

### 演示 Demo

我们用该模型做了一个医疗问答演示。将来，我们会将这款产品做成微信小程序与大家见面。

We made a demo of medical QA system with this model. In the future, we will make this product into a wechat app to meet you.

![avatar](https://huggingface.co/IDEA-CCNL/YuyuanQA-GPT2-3.5B/resolve/main/QA-DEMO.png)

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
