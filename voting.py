import pandas as pd
import numpy as np


def main():
    res1 = pd.read_csv('./test/all_in_new_select300.csv')
    res1.columns = ['cust_no', 'label1']
    res2 = pd.read_csv('./test/xgb_all_in.csv')
    res2.columns = ['cust_no', 'label2']
    res3 = pd.read_csv('./test/cat_all_in.csv')
    res3.columns = ['cust_no', 'label3']

    res1 = res1.merge(res2, on='cust_no', how='left')
    res1 = res1.merge(res3, on='cust_no', how='left')
    res1['label'] = res1[['label1', 'label2', 'label3']].mean(axis=1)  # 平均投票
    res1['label'] = res1['label'].apply(lambda x: round(x,0))
    res1['label'] = res1['label'].astype(int)
    res1[['cust_no', 'label']].to_csv('./test/voting.csv', index=None)

    res1['label'] = 0.5 * res1['label1'] + 0.3 * res1['label2'] + 0.2 * res1['label3']  # 权重投票
    res1['label'] = res1['label'].apply(lambda x: round(x,0))
    res1['label'] = res1['label'].astype(int)
    res1[['cust_no', 'label']].to_csv('./test/voting_weighted.csv', index=None)


if __name__ == "__main__":
    main()