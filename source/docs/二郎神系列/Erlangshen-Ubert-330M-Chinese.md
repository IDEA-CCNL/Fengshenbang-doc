# Erlangshen-Ubert-330M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

采用统一的框架处理多种抽取任务，AIWIN2022的冠军方案，3.3亿参数量的中文UBERT-Large。

Adopting a unified framework to handle multiple information extraction tasks, AIWIN2022's champion solution, Chinese UBERT-Large (330M).

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言理解 NLU | 二郎神 Erlangshen | UBERT |      330M      |    中文 Chinese    |

## 模型信息 Model Information

参考论文：[Unified BERT for Few-shot Natural Language Understanding](https://arxiv.org/abs/2206.12094)

UBERT是[2022年AIWIN世界人工智能创新大赛：中文保险小样本多任务竞赛](http://ailab.aiwin.org.cn/competitions/68#results)的冠军解决方案。我们开发了一个基于类似BERT的骨干的多任务、多目标、统一的抽取任务框架。我们的UBERT在比赛A榜和B榜上均取得了第一名。因为比赛中的数据集在比赛结束后不再可用，我们开源的UBERT从多个任务中收集了70多个数据集（共1,065,069个样本）来进行预训练，并且我们选择了[MacBERT-Large](https://huggingface.co/hfl/chinese-macbert-large)作为骨干网络。除了支持开箱即用之外，我们的UBERT还可以用于各种场景，如NLI、实体识别和阅读理解。示例代码可以在[Github](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/dev/yangping/fengshen/examples/ubert)中找到。

UBERT was the winner solution in the [2022 AIWIN ARTIFICIAL INTELLIGENCE WORLD INNOVATIONS: Chinese Insurance Small Sample Multi-Task](http://ailab.aiwin.org.cn/competitions/68#results). We developed a unified framework based on BERT-like backbone for multiple tasks and objectives. Our UBERT owns first place, as described in leaderboards A and B. In addition to the unavailable datasets in the challenge, we carefully collect over 70 datasets (1,065,069 samples in total) from a variety of tasks for open-source UBERT. Moreover, we apply [MacBERT-Large](https://huggingface.co/hfl/chinese-macbert-large) as the backbone. Besides out-of-the-box functionality, our UBERT can be employed in various scenarios such as NLI, entity recognition, and reading comprehension. The example codes can be found in [Github](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/dev/yangping/fengshen/examples/ubert).

- 论文：[Unified BERT for Few-shot Natural Language Understanding](https://arxiv.org/pdf/2206.12094.pdf)
- 知乎：[AIWIN大赛冠军，IDEA研究院封神榜提出多任务学习方案Ubert](https://zhuanlan.zhihu.com/p/539958182?)

## 使用 Usage

### 模型下载地址 Download Address
 
| 模型 | 地址   |
|:---------:|:--------------:|
| Erlangshen-Ubert-110M-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-110M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-110M-Chinese)       |
| Erlangshen-Ubert-330M-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-330M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-330M-Chinese)   |

### 加载模型 Loading Models

Pip install fengshen

```python
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable ./
```

Run the code

```python
import argparse
from fengshen import UbertPipelines

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UbertPipelines.pipelines_args(total_parser)
args = total_parser.parse_args()

args.pretrained_model_path = "IDEA-CCNL/Erlangshen-Ubert-330M-Chinese"

test_data=[
    {
        "task_type": "抽取任务", 
        "subtask_type": "实体识别", 
        "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。", 
        "choices": [ 
            {"entity_type": "小区名字"}, 
            {"entity_type": "岗位职责"}
            ],
        "id": 0}
]

model = UbertPipelines(args)
result = model.predict(test_data)
for line in result:
    print(line)
```

## Finetune 使用

开源的模型我们已经经过大量的数据进行预训练而得到，可以直接进行 Zero-Shot，如果你还想继续finetune,可以参考我们的 [example.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/ubert/example.py)。你只需要将我们数据预处理成为我们定义的格式，即可使用简单的几行代码完成模型的训练和推理。我们是复用 pytorch-lightning 的 trainer 。在训练时，可以直接传入 trainer 的参数，此外我们还定义了一些其他参数。常用的参数如下：


```sh
--pretrained_model_path       #预训练模型的路径，默认
--load_checkpoints_path       #加载模型的路径，如果你finetune完，想加载模型进行预测可以传入这个参数
--batchsize                   #批次大小, 默认 8
--monitor                     #保存模型需要监控的变量，例如我们可监控 val_span_acc
--checkpoint_path             #模型保存的路径, 默认 ./checkpoint
--save_top_k                  #最多保存几个模型, 默认 3
--every_n_train_steps         #多少步保存一次模型, 默认 100
--learning_rate               #学习率, 默认 2e-5
--warmup                      #预热的概率, 默认 0.01
--default_root_dir            #模型日子默认输出路径
--gradient_clip_val           #梯度截断， 默认 0.25
--gpus                        #gpu 的数量
--check_val_every_n_epoch     #多少次验证一次， 默认 100
--max_epochs                  #多少个 epochs， 默认 5
--max_length                  #句子最大长度， 默认 512
--num_labels                  #训练每条样本最多取多少个label，超过则进行随机采样负样本， 默认 10
```

## 数据预处理示例

整个模型的 Pipelines 我们已经写好，所以为了方便，我们定义了数据格式。目前我们在预训练中主要含有一下几种任务类型

| task_type | subtask_type   |
|:---------:|:--------------:|
| 分类任务  | 文本分类       |
|           | 自然语言推理   |
|           | 情感分析       |
|           | 多项式阅读理解 |
| 抽取任务  | 实体识别       |
|           | 事件抽取       |
|           | 抽取式阅读理解 |
|           | 关系抽取       |

### 分类任务

#### 普通分类任务
对于分类任务，我们把类别描述当作是 entity_type，我们主要关注 label 字段，label为 1 表示该该标签是正确的标签。如下面示例所示
```json
{
	"task_type": "分类任务",
	"subtask_type": "文本分类",
	"text": "7000亿美元救市方案将成期市毒药",
	"choices": [{
		"entity_type": "一则股票新闻",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "一则教育新闻",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "一则科学新闻",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}

```

#### 自然语言推理
```json
{
	"task_type": "分类任务",
	"subtask_type": "自然语言推理",
	"text": "在白云的蓝天下，一个孩子伸手摸着停在草地上的一架飞机的螺旋桨。",
	"choices": [{
		"entity_type": "可以推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "不能推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "很难推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}
```


#### 语义匹配

```json
{
	"task_type": "分类任务",
	"subtask_type": "语义匹配",
	"text": "不要借了我是试试看能否操作的",
	"choices": [{
		"entity_type": "不能理解为：借款审核期间能否取消借款",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "可以理解为：借款审核期间能否取消借款",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}

```

### 抽取任务
对于抽取任务，label 字段是无效的
#### 实体识别
```json
{
	"task_type": "抽取任务",
	"subtask_type": "实体识别",
	"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，",
	"choices": [{
		"entity_type": "地址",
		"label": 0,
		"entity_list": [{
			"entity_name": "台湾",
			"entity_type": "地址",
			"entity_idx": [
				[15, 16]
			]
		}]
	}{
		"entity_type": "政府机构",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "电影名称",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "人物姓名",
		"label": 0,
		"entity_list": [{
			"entity_name": "彭小军",
			"entity_type": "人物姓名",
			"entity_idx": [
				[0, 2]
			]
		}]
	},
	"id": 0
}

```
#### 事件抽取
```json

{
	"task_type": "抽取任务",
	"subtask_type": "事件抽取",
	"text": "小米9价格首降，6GB+128GB跌了200，却不如红米新机值得买",
	"choices": [{
		"entity_type": "降价的时间",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "降价的降价方",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "降价的降价物",
		"label": 0,
		"entity_list": [{
			"entity_name": "小米9",
			"entity_type": "降价的降价物",
			"entity_idx": [
				[0, 2]
			]
		}, {
			"entity_name": "小米9",
			"entity_type": "降价的降价物",
			"entity_idx": [
				[0, 2]
			]
		}]
	}, {
		"entity_type": "降价的降价幅度",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}
```
#### 抽取式阅读理解

```json
{
	"task_type": "抽取任务",
	"subtask_type": "抽取式阅读理解",
	"text": "截至2014年7月1日，圣地亚哥人口估计为1381069人，是美国第八大城市，加利福尼亚州第二大城市。它是圣迭戈-蒂华纳城市群的一部分，是美国与底特律-温莎之后的第二大跨境城市群，人口4922723。圣地亚哥是加州的出生地，以全年温和的气候、天然的深水港、广阔的海滩、与美国海军的长期联系以及最近作为医疗和生物技术发展中心而闻名。",
	"choices": [{
		"entity_type": "除了医疗保健，圣迭戈哪个就业部门已经强势崛起？",
		"label": 0,
		"entity_list": [{
			"entity_name": "生物技术发展",
			"entity_idx": [
				[153, 158]
			]
		}]
	}, {
		"entity_type": "在所有的军事部门中，哪一个在圣地亚哥的存在最为强大？",
		"label": 0,
		"entity_list": [{
			"entity_name": "美国海军",
			"entity_idx": [
				[135, 138]
			]
		}]
	}, {
		"entity_type": "在美国十大城市中，圣迭戈排名哪一位？",
		"label": 0,
		"entity_list": [{
			"entity_name": "第八",
			"entity_idx": [
				[33, 34]
			]
		}]
	}],
	"id": 0
}
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的对该模型的论文：

If you are using the resource for your work, please cite the our paper for this model:

```text
@article{fengshenbang/ubert,
  author    = {JunYu Lu and
               Ping Yang and
               Jiaxing Zhang and
               Ruyi Gan and
               Jing Yang},
  title     = {Unified {BERT} for Few-shot Natural Language Understanding},
  journal   = {CoRR},
  volume    = {abs/2206.12094},
  year      = {2022}
}
```

如果您在您的工作中使用了我们的模型，也可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [overview paper](https://arxiv.org/abs/2209.02970):

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
