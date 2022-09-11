import pandas as pd
import numpy as np

aspect_dic = {'service':'服務','location':'位置','value':'性價比','cleanliness':'乾淨','room':'客房','facilities':'設施'}
auxiliary_sentence_list = ['服務', '位置', '性價比', '乾淨', '客房', '設施']

def create_dataset_for_NLIM(df):
    # 將輔助句合併進df
    df['auxiliary_sentence'] = [auxiliary_sentence_list for i in df.index]
    df_2 = df.explode('auxiliary_sentence')
    print("原本的df:",df.shape)
    print("建構輔助句後:",df_2.shape)

    flag = 0
    check_column_list = ['service','location','value','cleanliness','room','facilities']
    label_id_list = [0, 1, 2] # 'none', 'positive', 'negative'
    output = pd.DataFrame()
    for i, g in df_2.groupby(np.arange(len(df_2)) // 6):
        flag = 0
        label_list = []
        for index, row in g.iterrows():
            #status = check_column_list[flag]
            status = row[check_column_list[flag]]
            #print(status)
            #print(g[status].iloc[0])

            if(status==0): #無包含
                label_list.append(0)
            elif(status==1): #有包含，正面情緒
                label_list.append(1)
            elif(status==2): #有包含，負面情緒
                label_list.append(2)
            #print(label_list)
            flag = flag+1

        dict_ = {'id':g.id,'review':g.reviews,'auxiliary_sentence':g.auxiliary_sentence,'label':label_list}
        #print(dict_)
        df_dictionary = pd.DataFrame.from_dict(dict_)
        output = pd.concat([output, df_dictionary], ignore_index=True)
    return output


# 引入原始資料集
path = 'reviewAspectSentiment/dataset/dataset(original)/reviews.csv'
df = pd.read_csv(path)

# 轉換資料集
df_convert = create_dataset_for_NLIM(df)

# 儲存檔案
df_convert.to_csv('reviewAspectSentiment/dataset/dataset(forModel)/dataset(forNLI-M).csv',encoding='utf_8_sig', index=False)


