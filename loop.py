import pandas as pd
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

# 進度條
from tqdm import tqdm
# 模型、特徵方法
from model import loss_function

def train_loop_function(data_loader, model, optimizer, device):
  """
  This function defines the training loop over the entire training set.
  """
  model.train()

  batch_count = 0
  running_loss = 0.0
  train_loss = 0.0
  corrects = 0
  total = 0
  for bi, d in enumerate(tqdm(data_loader)):
    
    ids = d["input_ids"]
    mask = d["attention_mask"]
    token_type_ids = d["token_type_ids"]
    targets = d["targets"]

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.long)

    optimizer.zero_grad()

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    
    loss = loss_function(outputs, targets)

    loss.backward()
    optimizer.step()

    #計算loss
    running_loss += loss.item()

    #計算accuracy
    _, predicted = torch.max(outputs, 1)
    total = total + targets.size(0)
    corrects = corrects + (predicted==targets).sum().item()

    batch_count += 1
  # loss  
  train_loss = running_loss / batch_count
  print("batch_count:",batch_count)

  # accuracy
  train_accuracy = corrects / total * 100
  print("total:",total)
  return train_accuracy, train_loss


def eval_loop_function(data_loader, model, device):
  """
  This function defines the evaluation loop over the entire validation set.
  """
  
  model.eval()

  corrects = 0
  total = 0
  batch_count = 0
  running_loss = 0.0
  val_loss = 0.0

  # 計算AUC
  prob_all = []
  lable_all = []
  
  for bi, d in enumerate(tqdm(data_loader)):
    #print("eval:",bi)
    ids = d["input_ids"]
    mask = d["attention_mask"]
    token_type_ids = d["token_type_ids"]
    targets = d["targets"]

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    
    # 計算loss
    loss = loss_function(outputs, targets)
    running_loss += loss.item()

    # 計算accuracy
    _, predicted = torch.max(outputs, 1)
    total = total + targets.size(0)
    corrects = corrects + (predicted==targets).sum().item()


    batch_count += 1
  
  # loss
  val_loss = running_loss / batch_count
  # accuracy
  val_accuracy = corrects / total * 100

  print("val accuracy:",val_accuracy)

  return val_accuracy, val_loss

def infer_loop_function(data_loader, model, device):
  """
  This function performs the inference on testing sets and stores the predicted
  values.
  """

  model.eval()

  df_pred = pd.DataFrame({"id": [], "predicted": [], "actual": []})

  ii = 0
  for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), ncols=80):
    opinions_id = d["opinions_id"]
    ids = d["input_ids"]
    mask = d["attention_mask"]
    token_type_ids = d["token_type_ids"]
    targets = d["targets"]

    opinions_id = opinions_id.to(device, dtype=torch.long)
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    _, predicted = torch.max(outputs, 1)
    
    predicted = predicted.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    opinions_id = opinions_id.detach().cpu().numpy()

    for k in range(len(predicted)):
      df_pred.loc[ii] = [str(opinions_id[k]), str(predicted[k]), str(targets[k])]
      ii += 1

    df_pred.to_csv('reviewAspectSentiment/results/PredictedValues.csv', index=False) # 測試結果存檔