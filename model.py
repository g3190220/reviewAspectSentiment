import os
import pandas as pd
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
# bert
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# 建構輸入特徵(for訓練、驗證)
class Build_features:
  """
  此類使用預訓練的 BERT tokenizer
  對輸入文本進行分詞，並返回相應的張量。
  """
  
  def __init__(self, text, auxiliary_sentence, targets, tokenizer, max_len):
    self.text = text
    self.auxiliary_sentence = auxiliary_sentence
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.targets = targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    text = str(self.text[item])
    auxiliary_sentence = str(self.auxiliary_sentence[item])
    targets = self.targets[item]

    inputs = self.tokenizer.encode_plus(
        text,auxiliary_sentence, # 句子對輸入
        add_special_tokens = True,
        max_length = self.max_len,
        padding = 'max_length',
        truncation = 'longest_first'
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long), #對應於文本序列中每個token的索引（在vocab中的索引）
        "attention_mask": torch.tensor(mask, dtype=torch.long), #對應於注意力機制的計算，各元素的值為0或1，如果當前token被mask或者是只是用來作為填充的元素，那麼其不需要進行注意力機制的計算，其值為0
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long), #對應於不同的文本序列，例如在NSP（BERT及某些語言模型中的“Next Sentence Prediction”）任務中需要輸入兩個文本序列
        "targets": torch.tensor(targets, dtype=torch.long)
    }

# 建構輸入特徵 (for 測試)
class Build_features_test:
  """
  此類使用預訓練的 BERT tokenizer
  對輸入文本進行分詞，並返回相應的張量。
  """
  
  def __init__(self, opinions_id, text, auxiliary_sentence, targets, tokenizer, max_len):
    self.opinions_id = opinions_id
    self.text = text
    self.auxiliary_sentence = auxiliary_sentence
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.targets = targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    opinions_id = self.opinions_id[item]
    text = str(self.text[item])
    auxiliary_sentence = str(self.auxiliary_sentence[item])
    targets = self.targets[item]

    #text = text + ' ' + auxiliary_sentence

    inputs = self.tokenizer.encode_plus(
        text,auxiliary_sentence,
        add_special_tokens = True,
        max_length = self.max_len,
        padding = 'max_length',
        truncation = 'longest_first'
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "opinions_id": torch.tensor(opinions_id, dtype=torch.long)
    }

# 模型架構
class SentimentClassifier(nn.Module):
  """
  此類定義模型架構
  它只是在預訓練的 BERT 模型之上，串接一個 fully-connected layer
  """

  def __init__(self, BERT_MODEL):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, 3) # Number of output classes = 3 (0：)

  def forward(self, ids, mask, token_type_ids):
    last_hidden_state, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
    output = self.drop(pooled_output)
    return self.out(output)


# 損失函數
def loss_function(outputs, targets):
	"""
	該函數定義了用於訓練模型的損失函數，即 CrossEntropy
	"""
	return nn.CrossEntropyLoss(reduction='mean')(outputs, targets)

# 計算class weight (處理類別不平衡問題)
def compute_class_weight(df):
  class_counts = []
  for i in range(3): # label為0、1、2
    class_counts.append(df[df['label']==i].shape[0])
  num_samples = sum(class_counts)
  labels = df['label'].values
  class_weights = []
  for i in range(len(class_counts)):
    if class_counts[i] != 0:
      class_weights.append(num_samples/class_counts[i])
    else:
      class_weights.append(0)
  weights = [class_weights[labels[i]] for i in range(int(num_samples))]
  sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
  return sampler
