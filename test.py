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
from model import Build_features_test
# 測試Loop方法
from loop import infer_loop_function
# 評估方法
from evaluate import aspect_sentiment_evaluation, aspect_category_evaluations

# 設置參數
BERT_MODEL = 'bert-base-chinese' #使用的預訓練模型
TEST_MAX_LEN = 160
TEST_BATCH_SIZE = 2

# 隱藏 error log
transformers.logging.set_verbosity_error()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.current_device()

# 確認是否可用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載預訓練tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL)

# 載入測試集
df_test = pd.read_csv('reviewAspectSentiment/dataset/test.csv', encoding='utf-8-sig')

"""
建構輸入
"""
# 建構特徵
test_dataset = Build_features_test(
      opinions_id = df_test['id'].values,
      text = df_test['review'].values,
      auxiliary_sentence = df_test['auxiliary_sentence'],
      targets = df_test['label'].values,
      tokenizer = tokenizer,
      max_len = TEST_MAX_LEN
)
print(f"Test Set: {len(test_dataset)}")

test_data_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size = TEST_BATCH_SIZE,
      shuffle = False
)

"""
載入模型
"""
model = torch.load('reviewAspectSentiment/models/2.bin')

"""
開始測試
"""
infer_loop_function(data_loader=test_data_loader, model=model, device=device)

"""
印出評估結果
"""
# 載入測試結果
df_predict = pd.read_csv('reviewAspectSentiment/results/PredictedValues.csv')

# 評估 aspect-sentiment
aspect_sentiment_evaluation(df_predict)

# 評估每一個 aspect (不包含情緒)
aspect_category_evaluations(df_predict)



