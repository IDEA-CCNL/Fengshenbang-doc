## 余元系列
医学领域的余元系列，35亿参数余元-3.5B大模型，采用50G的医疗领域数据和知识，在已有的通用模型基础上继续训练，32张A100训练7天，是目前最大的开源GPT2医疗大模型。我们的模型在医学领域的事实判断中具有近90%的准确率。

我们利用余元-3.5B大模型实现事实判断，医学问答。更多的可能性等着你去发现。


### 模型下载地址
[Huggingface 余元-3.5B](https://huggingface.co/IDEA-CCNL/Yuyuan-GPT2-3.5B)

### load model
```python 
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Yuyuan-GPT2-3.5B')
model = GPT2Model.from_pretrained('IDEA-CCNL/Yuyuan-GPT2-3.5B')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
### generation
```python
from transformers import pipeline, set_seed
set_seed(55)
generator = pipeline('text-generation', model='IDEA-CCNL/Yuyuan-GPT2-3.5B')
generator("Diabetics should not eat", max_length=30, num_return_sequences=1)

```