## Randeng-PPVAE-1.2B-General-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/models/PPVAE)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/Randeng-PPVAE-1.2B-General-Chinese.html)

## 简介 Brief Introduction

PPVAE(Pre-train and Plug-in Variational Auto-Encoder) 可以通过少量类别文本的训练生成大量该类别的增强样本。
PPVAE是一个由两个VAE组成的层级框架：预训练VAE的编码器得到文本全局隐空间，解码器将隐向量解码为文本；PluginVAE为一个轻量级VAE，学习从全局隐空间到条件隐空间的相互映射，该映射只需要少量条件文本即可训练完成。

PPVAE (Pre-train and Plug-in Variational Auto-Encoder) can generate a large number of category-specific samples from the training of a small number of category texts. PPVAE is a hierarchical framework consisting of two VAEs: the encoder of the pre-trained VAE gets the text global hidden space and the decoder decodes the hidden vector into text; PluginVAE is a lightweight VAE that learns the transformation from the global hidden space to the conditional hidden space, which requires only a small amount of conditional text to be trained.

PPVAE参考论文[Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders](https://arxiv.org/abs/1911.03882).

PPVAE reference paper [Pre-training and Plug-in: Flexible Conditional Text Generation with Variable Autoencoders](https://arxiv.org/abs/1911.03882).

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 数据增强 Augmentation | 自然语言生成 NLG | 燃灯 Randeng | VAE |      1.2B      |     pluginVAE    |

## 模型信息 Model Information

**Pretrained VAE:**

训练语料：悟道语料库（280G版本）

Training Corpus: Wudao Corpus (with 280G samples)

参考模型：[Randeng-DAVAE-1.2B-General-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese)

Reference model:[Randeng-DAVAE-1.2B-General-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese)

**PluginVAE:**

编码器：三层MLP，将隐向量从全局隐空间映射到类别隐空间；

解码器：三层MLP，将隐向量从类别隐空间映射到全局隐空间。

训练语料：少量类别文本。

Encoder: three-layer MLP that maps the hidden vector from the global hidden space to the category hidden space.

Decoder: three-layer MLP, mapping hidden vectors from the category hidden space to the global hidden space.

Training corpus: a small amount of categorical text.

## 使用 Usage

```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable .
```

```python3
import torch
from transformers import BertTokenizer,T5Tokenizer
from fengshen.models.PPVAE.pluginVAE import PPVAEModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_texts = [
    "非常好的一个博物馆，是我所有去过的博物馆里感觉最正规的一家.", 
    "这是我来长沙最最期待的一定要去的地方，总算今天特地去瞻仰千古遗容了，真好。", 
    "地方很大 很气派~~可以逛很久~~~去的时候是免费的~不过要安检~~~里面的马王堆~幸追夫人~还是很不错的",
    "绝对不虚此行！相当震撼的展览！原来古人也化妆，还有假发。记忆最深的是那个藕汤。可惜真颜已不得见。", 
    "去过三次，个人认为这是长沙最值得去的地方.", 
    "非常喜欢的一家博物馆，里面可看的东西很多，当然最吸引我的就是那个辛追夫人和“素纱单衣”，果然不是盖的~赞~~~", 
    "这两年也有很多机会去博物馆。。。不过还是想说湖南省博物馆是非常有特色的。。。真是上了一节很生动的历史课。",
    "网上订票去的，还是很顺利的就进去了，里面挺清净的，外围的环境也不错，还有鸽子可以喂。",
]
encoder_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Randeng-PPVAE-1.2B-General-Chinese")
decoder_tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-PPVAE-1.2B-General-Chinese", eos_token = '<|endoftext|>', pad_token = '<pad>',extra_ids=0)
decoder_tokenizer.add_special_tokens({'bos_token':'<bos>'})
ppvae_model = PPVAEModel.from_pretrained("IDEA-CCNL/Randeng-PPVAE-1.2B-Augmentation-Chinese").to(device)
ppvae_model.train_plugin(encoder_tokenizer,decoder_tokenizer,input_texts,negative_samples=None)
# n:输出样本数量
texts = ppvae_model.generate(n=5)
print(texts)
# 生成结果样例：
# ['同学很推荐那里,自然会有好的风景.那里物价很便宜,真的不错。', 
# '同学说一会去盛国,可能是我去的比较多!故居真的很漂亮,夜景也特别好看。'
# '我的第一次旅行没有白来,最后领略了有些风吹草低见牛羊的味道,谢谢本次疗养。', 
# '同学一打听:这里距离世纪公园,还有最近的香山营不过200米,海拔也才四千米。', 
# '我发现那边很文艺!!有机会去过的,真是土耳其当地口音~还是很干净!。', ]

```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

If you are using the resource for your work, please cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```