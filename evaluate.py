from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score

def get_confusion_matrix(df):
    return confusion_matrix(df['actual'],df['predicted'])

def get_classification_report(df):
    return classification_report(df['actual'],df['predicted'])

def get_accuracy(df):
    return accuracy_score(df['actual'],df['predicted'])

def get_macro_f1(df):
    return f1_score(df['actual'],df['predicted'], average='macro')

def get_auc(df):
    return roc_auc_score(df['actual'],df['predicted'])

def aspect_sentiment_evaluation(df):
    print(get_confusion_matrix(df))
    print(get_classification_report(df))
    print("Accuracy: ", get_accuracy(df))
    print("Macro F1: ", get_macro_f1(df))

def aspect_category_evaluation(aspect, df):
    """
    評估每個aspect分類的結果
    """
    
    fliter = (df['aspect'] == aspect)
    df_cut = df[fliter]

    print("Aspect: "+ aspect)
    print(get_confusion_matrix(df_cut))
    print(get_classification_report(df_cut))
    print("AUC: ", get_auc(df_cut))
    print("Macro F1: ", get_macro_f1(df_cut))
    print("-----------------------------------------------------------------------")

def aspect_category_evaluations(df):
    """
    此方法評估每個aspect分類的結果(非情緒，單純只aspect分類)
    => label有包含(1)、無包含(0)
    做法：轉換原本資料集的label
        - 1(正面)、2(負面) => 1(有包含) 
        - 0(無包含) => 0(無包含)
    """
    # 轉換label
    df = df.replace([1, 2], 1).replace(0, 0) 
    # 加各aspect
    aspect_list = ['service','location','value','cleanliness','room','facilities']
    df['aspect'] = aspect_list*int((df.shape[0]/6))
    
    for item in aspect_list:
        aspect_category_evaluation(item, df)
    