# Bigan-Transformer-XL-denoise-1.1B

Bigan-Transformer-XL-denoise-1.1B是在基于清华大学开源的[Chinese-Transformer-XL](https://github.com/THUDM/Chinese-Transformer-XL)基础上，使用180G的中文数据进行prompt+denoise目标的增量预训练得到的模型。


## 任务描述

Bigan-Transformer-XL-denoise-1.1B在[Chinese-Transformer-XL](https://github.com/THUDM/Chinese-Transformer-XL)的基础上，在180G的中文通用预训练语料的基础上进行继续训练。去噪的任务受到了[BART](https://arxiv.org/abs/1910.13461)和[EDA](https://arxiv.org/abs/1901.11196)的启发，一共包括以下几种：
1. 增：增加来自同文本其他位置的字词，随机字词或者当前字符的重复。
2. 删：随机根据比例删除文段里的字词。
3. 换：随机根据比例交换文段里的一对字词位置。
4. 改：随机根据比例将文段里的字词改为随机字词。
5. 乱：随机打乱文段里的句子顺序。
数据处理好后会以提示模版(prompt template)的形式输入transformer-xl。具体模版形式为：f"“{noisy_sent}”改写后是“", f"{denoise_sent}”。"

## 数据处理

通用的数据举例：
```
text: '本文讨论了α-混合序列的等价条件,对混合速度要求低,改进了相关结果.'
```
add noise后的训练数据举例(注意符号为全角符号)
```
noisy_input: '本文讨论了α-无可混合序列的等价条件▁不久,对混合年世界速度要求低,改进▁为了相关结果1956.'

model_input: “本文讨论了α-无可混合序列的等价条件▁不久,对混合年世界速度要求低,改进▁为了相关结果1956.”去噪后是“本文讨论了α-混合序列的等价条件,对混合速度要求低,改进了相关结果.”
```


## 模型使用

```python
from fengshen.models.transfo_xl_denoise.tokenization_transfo_xl_denoise import TransfoXLDenoiseTokenizer
from fengshen.models.transfo_xl_denoise.modeling_transfo_xl_denoise import TransfoXLDenoiseModel
from fengshen.models.transfo_xl_denoise.generate import denoise_generate

tokenizer = TransfoXLDenoiseTokenizer.from_pretrained('IDEA-CCNL/Bigan-Transformer-XL-denoise-1.1B')
model = TransfoXLDenoiseModel.from_pretrained('IDEA-CCNL/Bigan-Transformer-XL-denoise-1.1B')

input_text = "凡是有成就的人, 都很严肃地对待生命自己的"
res = denoise_generate(model, tokenizer,  input_text)
print(res) # "有成就的人都很严肃地对待自己的生命。"
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