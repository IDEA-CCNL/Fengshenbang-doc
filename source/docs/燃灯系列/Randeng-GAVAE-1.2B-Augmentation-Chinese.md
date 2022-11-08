# Randeng-GAVAE-1.2B-General-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/models/GAVAE)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/Randeng-GAVAE-1.2B-General-Chinese.html)

## 简介 Brief Introduction

GAVAE(Generative Adversarial Varational Auto-encoder)在预训练VAE模型的隐空间插入GAN网络，对类别文本的隐向量进行对抗生成训练，以少量特定类别文本训练后即可生成该类别文本。

GAVAE (Generative Adversarial Varational Auto-encoder) inserts a GAN model into the hidden space of the pre-trained VAE model and performs generative Adversarial training on the hidden vectors of a small number of categories of text, which can be used to generate that category of text after training.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 数据增强 Augmentation | 自然语言生成 NLG | 燃灯 Randeng | VAE |      1.2B      |     GAN   |

## 模型信息 Model Information

**Pretrained VAE:**

训练语料：悟道语料库（280G版本）

Training Corpus: Wudao Corpus (with 280G samples)

参考模型：[Randeng-DAVAE-1.2B-General-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese)

Reference model:[Randeng-DAVAE-1.2B-General-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese)

**GAN**

生成器：五层MLP，生成类别隐向量；

判别器：三层MLP，判断向量为真实类别隐向量或生成器生成的向量。

训练语料：少量类别文本。

Generator: five-layer MLP, generating category hidden vectors.

Discriminator: three-layer MLP, which determines whether the vector is a true category hidden vector or a generated vector.

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
from fengshen.models.GAVAE.GAVAEModel import GAVAEModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_texts = [
    "非常好的一个博物馆，是我所有去过的博物馆里感觉最正规的一家，凭有效证件可以入馆，可以自助免费存小件物品，讲解员和馆内外的工作人员也非常认真，其他的服务人员也很热情，非常好的！馆内的藏品也让人非常震撼！希望继续保持～", 
    "这是我来长沙最最期待的一定要去的地方，总算今天特地去瞻仰千古遗容了，开车到门口大屏幕显示着门票已发完的字样，心里一惊以为今天是白来了。但进了停车场才知道凭停车卡和有效身份证里面也能领，停车还不花钱，真好。", 
    "地方很大 很气派~~可以逛很久~~~去的时候是免费的~不过要安检~~~里面的马王堆~幸追夫人~还是很不错的~~~~去的时候有一个吴越文化特别展~~~东西也很多~~~~~很好看",
    "我们到达的时候是下午3点，门票已经发完了。当时正焦虑的不知道怎么办才好，门卫大哥给我们俩补办了门票，这才得以入馆。非常感谢！绝对不虚此行！相当震撼的展览！原来古人也化妆，还有假发。记忆最深的是那个藕汤。可惜真颜已不得见。", 
    "去过三次，个人认为这是长沙最值得去的地方，博物馆的重点就是辛追，遗憾的是，每次去我都会感到悲哀，虽然我三次去的时候都要门票，但是每次看到辛追，都觉得现代的人类不应该挖她出来，除了第一次我觉得辛追像刚死去一样，后来两次我觉得太惨不忍睹了。建议大家要去就早去，以后肯定越来越腐烂", 
    "上大学时候去的，当时学生证是半价25，后来凭有效证件就不要钱了。非常喜欢的一家博物馆，里面可看的东西很多，当然最吸引我的就是那个辛追夫人和“素纱单衣”，果然不是盖的~里面的讲解员大部分都是师大学历史类的，非常专业和有耐心。虽然不在长沙了，不过对那里还是很有感情的，赞~~~", 
    "这两年也有很多机会去博物馆。。。不过还是想说湖南省博物馆是非常有特色的。。。应该说整个展览分成两个部分吧。。。一个部分是马王堆的主体展。。。另一个就是湖南的一些考古发现。。。其实来省博大部分的游客还是冲着马王堆来的吧。。。博物馆也很有心的为每一批游客安排了讲解员。。。从马王堆的发现到马王堆出土文物的介绍再到最后棺木和辛追的介绍。。。真是上了一节很生动的历史课。",
    "网上订票去的，还是很顺利的就进去了，里面挺清净的，外围的环境也不错，还有鸽子可以喂。那天不是很闹，兜了一圈感觉还是很顺畅的，老娘娘和金缕玉衣挺震撼的。到此一游还是挺需要的",
]
encoder_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Randeng-GAVAE-1.2B-Augmentation-Chinese")
decoder_tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-GAVAE-1.2B-Augmentation-Chinese", eos_token = '<|endoftext|>', pad_token = '<pad>',extra_ids=0)
decoder_tokenizer.add_special_tokens({'bos_token':'<bos>'})
gavae_model = GAVAEModel.from_pretrained("IDEA-CCNL/Randeng-GAVAE-1.2B-Augmentation-Chinese").to(device)
gavae_model.train_gan(encoder_tokenizer,decoder_tokenizer,input_texts)
# n:输出样本数量
texts = gavae_model.generate(n=5)
print(texts)

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