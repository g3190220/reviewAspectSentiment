import pandas as pd
import numpy as np

# 計算某個Aspect的個數
def caculate_aspect_sentiment_score(aspect_):
  positive_count = 0
  negative_count = 0
  
  positive_count = (aspect_['predicted'] == 1).sum() # 非負面個數
  negative_count = (aspect_['predicted'] == 2).sum() # 負面個數

  return [positive_count, negative_count]


# 得到某個資料集下，每個Aspect的情緒個數
def create_sentiment_score(df_):
    aspect_list = ['服務', '位置', '性價比', '乾淨', '客房', '設施']
    sentiment_list = ['非負面','負面']
    df_['aspect'] = np.resize(aspect_list,len(df_))
    score_df = pd.DataFrame(columns=aspect_list,index=sentiment_list)
    for aspect in aspect_list:
        score_list = caculate_aspect_sentiment_score(df_[df_['aspect']==aspect])
        print(aspect)
        print(score_list)
        score_df[aspect] = score_list
    return score_df

if __name__ == '__main__':
    # 載入資料集 (放入模型產出的測試結果資料)
    df = pd.read_csv('reviewAspectSentiment/results/PredictedValues.csv')
    # 計算所有情緒面向的評論個數
    df2 = create_sentiment_score(df)
    # 存到scores的資料夾中
    df2.to_csv('reviewAspectSentiment/scores/sentiment_score.csv')
    

