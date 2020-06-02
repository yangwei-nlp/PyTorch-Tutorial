"""讲述transformers包中tokenizer的使用"""
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
token_ids = tokenizer.encode('这是一句话', max_length=256, pad_to_max_length=True)
print(2333333)
