## Randeng-DAVAE-1.2B-General-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/models/DAVAE)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/Randeng-DAVAE-1.2B-General-Chinese.html)

## 简介 Brief Introduction

使用101M的Bert作为encoder，1.1B参数量的transformer-XL作为decoder，以此构成变分自编码(VAE)网络。在训练中为了得到更好的潜在表示，对输入的embedding施加连续gaussian noise并且使用对抗学习训练后验网络，这就是DAVAE的由来。

The Variational Autoencoder (VAE) network comprises an encoder using Bert with 101M parameters and a decoder using transformer-XL with 1.1B parameters. To make the representation more expressive, the input embedding is perturbed with gaussian noise, and adversarial learning is used to train the posterior network, so forming the DAVAE.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言生成 NLG | 燃灯 Randeng | VAE |      1.2B      |     连续潜在空间 Continuous latent space    |

## 模型信息 Model Information

**数据准备 Corpus Preparation**

* 悟道语料库（280G版本）

* Wudao Corpus (with 280G samples) 

**防止后验崩塌 Avoiding posterior collapse**

为了防止在通用语料上后验崩塌，我们在训练中加入以下措施，
1. 使用KL annealing。对于正则项系数，采用梯形scheduler
2. 加入free bits。设置free bits，避免过分靠近先验
3. 强化潜在向量引导。潜在空间向量和decoder隐层输出逐位相加
4. 在输入embedding上加入连续gaussian噪声，与之前工作使用离散加噪方式不同
5. 在潜在空间进行对抗训练


We used several methods to avoid posterior collapse, as what follows,
1. Using KL annealing. A trapezoidal scheduler was used to calculate the coefficient for the regularization term. 
2. Adding free-bits constraint. we chose a certain free bit to avoid getting too close to the prior in the training.
3. Strengthening the guidance of the latent vector. The latent vector was added over the hidden state of every token.
4. Adding gaussian noise to the input embedding, differing from the noising method used in previous work.
5. Adversarial training in latent space.


## 使用 Usage

```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable .
```

```python3
import torch
from fengshen.models.DAVAE.DAVAEModel import DAVAEModel
from transformers import BertTokenizer,T5Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese")
decoder_tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese", eos_token = '<|endoftext|>', pad_token = '<pad>',extra_ids=0)
decoder_tokenizer.add_special_tokens({'bos_token':'<bos>'})
vae_model = DAVAEModel.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese").to(device)
input_texts = [
    "针对电力系统中的混沌振荡对整个互联电网的危害问题,提出了一种基于非线性光滑函数的滑模控制方法.",
    "超市面积不算大.挺方便附近的居民购买的. 生活用品也比较齐全.价格适用中.",
]
output_texts = vae_model.simulate_batch(encoder_tokenizer,decoder_tokenizer,input_texts)
print(output_texts)


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