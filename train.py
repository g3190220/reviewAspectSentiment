import os
import pandas as pd
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
# bert
import transformers
# 模型、特徵方法
from model import Build_features, SentimentClassifier, compute_class_weight
# 訓練、驗證Loop方法
from loop import train_loop_function, eval_loop_function

# 設置訓練參數
BERT_MODEL = 'bert-base-chinese' #使用的預訓練模型
TRAIN_MAX_LEN = 160
VALID_MAX_LEN = 160
TRAIN_BATCH_SIZE = 2 #出現cuda run time error => 改小
VALID_BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 3e-5

# 隱藏 error log
transformers.logging.set_verbosity_error()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.current_device()

# 確認是否可用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載預訓練tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL)

# 載入訓練和驗證集
df_train = pd.read_csv('reviewAspectSentiment/dataset/train.csv', encoding='utf-8-sig')
df_val = pd.read_csv('reviewAspectSentiment/dataset/val.csv', encoding='utf-8-sig')

"""
建構輸入
"""
# (1) 訓練集打亂
df_train_shuffle = df_train.sample(frac=1).reset_index(drop=True)

# (2) 自定義 sampler 以補償訓練集中的類別不平衡，只用在訓練集
sampler = compute_class_weight(df_train_shuffle)

# (3) 建構特徵
train_dataset = Build_features(
  text = df_train_shuffle['review'].values,
  auxiliary_sentence = df_train_shuffle['auxiliary_sentence'].values,
  targets = df_train_shuffle['label'].values,
  tokenizer = tokenizer,
  max_len = TRAIN_MAX_LEN
)

valid_dataset = Build_features(
      text = df_val['review'].values,
      auxiliary_sentence = df_val['auxiliary_sentence'],
      targets = df_val['label'].values,
      tokenizer = tokenizer,
      max_len = VALID_MAX_LEN
)

# (4) 創建dataloader
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = TRAIN_BATCH_SIZE,
    shuffle = False,
    sampler = sampler
)

valid_data_loader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size = VALID_BATCH_SIZE,
      shuffle = False
)

"""
載入預訓練模型
"""
model = SentimentClassifier(BERT_MODEL)
model = model.to(device)

"""
模型iteration、optimizer、scheduler設置
"""
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler
scheduler = lr_scheduler.StepLR(
      optimizer,
      step_size = 1,
      gamma = 0.8
)

"""
開始訓練
"""
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
print(f"use {device} to train...\n")
for epoch in range(EPOCHS):
    print("epoch:",epoch)
    train_accuracy, train_loss = train_loop_function(data_loader=train_data_loader, model=model, optimizer=optimizer, device=device)
    val_accuracy, val_loss = eval_loop_function(data_loader=valid_data_loader, model=model, device=device)

    print(f"\nEpoch = {epoch}\tAccuracy Score = {val_accuracy}")
    print(f"Learning Rate = {scheduler.get_last_lr()[0]}\n")

    scheduler.step()

    # 存檔
    path = 'reviewAspectSentiment/models/'+str(epoch)+'.bin'
    torch.save(model,path)

