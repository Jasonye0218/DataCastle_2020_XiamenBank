import pandas as pd
import numpy as np
import warnings
import os
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')


def gen_pesudo_data():
    tmp_aum = pd.read_csv('./train/aum/aum_m10.csv')
    tmp_aum[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']] = np.nan
    tmp_aum.to_csv('./train/aum/aum_m6.csv', index=None)
    tmp_cunkuan = pd.read_csv('./train/cunkuan/cunkuan_m10.csv')
    tmp_cunkuan[['C1', 'C2']] = np.nan
    tmp_cunkuan.to_csv('./train/cunkuan/cunkuan_m6.csv', index=None)
    tmp_beh = pd.read_csv('./train/behavior/behavior_m9.csv')
    tmp_beh[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']] = np.nan
    tmp_beh.to_csv('./train/behavior/behavior_m6.csv', index=None)


class GetData(object):

    def load_aum_data(self, path='./', m=1, iftrain=True):
        data = pd.read_csv(path + 'aum_m{}.csv'.format(m))
        m = m if iftrain else m + 12
        data.columns = ['cust_no', 'X1_{}'.format(m), 'X2_{}'.format(m), 'X3_{}'.format(m), 'X4_{}'.format(m),
                        'X5_{}'.format(m), 'X6_{}'.format(m), 'X7_{}'.format(m), 'X8_{}'.format(m)]
        return data

    def load_cunkuan_data(self, path='./', m=1, iftrain=True):
        data = pd.read_csv(path + 'cunkuan_m{}.csv'.format(m))
        data['C3'] = data['C1'] / (1e-3 + data['C2'])  # 平均理财产品金额
        m = m if iftrain else m + 12
        data.columns = ['cust_no', 'C1_{}'.format(m), 'C2_{}'.format(m), 'C3_{}'.format(m)]
        return data

    def load_beh_data(self, path='./', m=1, iftrain=True):
        data = pd.read_csv(path + 'behavior_m{}.csv'.format(m))
        m = m if iftrain else m + 12
        data['B8'] = data['B3'] / (1e-3 + data['B2'])  # 平均转入金额
        data['B9'] = data['B5'] / (1e-3 + data['B4'])  # 平均转出金额
        if m in [3, 6, 9, 12, 15]:
            data['x'] = pd.to_datetime(data['B6']) + pd.tseries.offsets.QuarterEnd()
            data['B6'] = (pd.to_datetime(data['x']) - pd.to_datetime(data['B6'])).dt.days  # 该季度与最后一次交易时间距离的天数
            data = data.drop('x', axis=1)
            data.columns = ['cust_no', 'B1_{}'.format(m), 'B2_{}'.format(m), 'B3_{}'.format(m), 'B4_{}'.format(m),
                            'B5_{}'.format(m), 'B6_{}'.format(m), 'B7_{}'.format(m), 'B8_{}'.format(m), 'B9_{}'.format(m)]
        else:
            data['B6'] = np.nan
            data['B7'] = np.nan
            data.columns = ['cust_no', 'B1_{}'.format(m), 'B2_{}'.format(m), 'B3_{}'.format(m), 'B4_{}'.format(m),
                            'B5_{}'.format(m), 'B6_{}'.format(m), 'B7_{}'.format(m), 'B8_{}'.format(m), 'B9_{}'.format(m)]
        return data

    def load_big_event_data(self,path='./', Q=1):
        data = pd.read_csv(path + 'big_event_Q{}.csv'.format(Q))
        cols_list = data.columns.to_list()
        cols_list = [x for x in cols_list if x not in ['cust_no', 'E15', 'E17']]  # E15, E17为金额，其他为日期
        for col in cols_list:
            if Q == 1:
                data[col] = (datetime(2020, 3, 31) - pd.to_datetime(data[col])).dt.days  # 转化为到季度末天数
            elif Q == 4:
                data[col] = (datetime(2019, 12, 31) - pd.to_datetime(data[col])).dt.days
            elif Q == 3:
                data[col] = (datetime(2019, 9, 30) - pd.to_datetime(data[col])).dt.days
        data.loc[data['E1'] > 20000, ['E1']] = np.nan  # 处理E1异常值
        return data

    def load_info_data(self,path='./', Q=1):
        data = pd.read_csv(path + 'cust_info_q{}.csv'.format(Q))
        return data


def get_data(data_file='', param='', Q=1):
    df_aum = pd.DataFrame()
    load_mon_list = ['load_aum_data', 'load_cunkuan_data', 'load_beh_data']
    if param in load_mon_list:   # 读取月份数据
        for mon in range(3, 0, -1):
            df = getattr(GetData(), param)(path='./test/{}/'.format(data_file), m=mon, iftrain=False)
            if len(df_aum):
                df_aum = df_aum.merge(df, on='cust_no', how='left')
            else:
                df_aum = df
        for mon in range(12, 5, -1):
            df = getattr(GetData(), param)(path='./train/{}/'.format(data_file), m=mon, iftrain=True)
            df_aum = df_aum.merge(df, on='cust_no', how='left')
    else:   # 读取季度数据
        if Q == 1:
            df_aum = getattr(GetData(), param)(path='./test/{}/'.format(data_file), Q=Q)
        else:
            df_aum = getattr(GetData(), param)(path='./train/{}/'.format(data_file), Q=Q)
    return df_aum


def get_quater_data(df, colhead='X', num_col=3, quater=3):
    columns = ['cust_no']
    rename_col = ['cust_no']
    for i in range(1, num_col + 1):
        for j in range(4):
            columns.append('{}{}_{}'.format(colhead, i, quater * 3 - j))  # 读取当前季度和前一季度最后一月数据
            rename_col.append('{}{}_{}'.format(colhead, i, 4 - j))
    tmp = df[columns]
    tmp.columns = rename_col
    return tmp


def statistics_feature_aum(df):
    for i in range(1, 9):
        df['X{}_mean'.format(i)] = df[['X{}_4'.format(i), 'X{}_3'.format(i), 'X{}_2'.format(i)]].mean(axis=1)
        df['X{}_std'.format(i)] = df[['X{}_4'.format(i), 'X{}_3'.format(i), 'X{}_2'.format(i)]].std(axis=1)
        df['X{}_sum'.format(i)] = df[['X{}_4'.format(i), 'X{}_3'.format(i), 'X{}_2'.format(i)]].sum(axis=1)
        df['X{}_max'.format(i)] = df[['X{}_4'.format(i), 'X{}_3'.format(i), 'X{}_2'.format(i)]].max(axis=1)
        df['X{}_min'.format(i)] = df[['X{}_4'.format(i), 'X{}_3'.format(i), 'X{}_2'.format(i)]].min(axis=1)

        for t in ['sub', 'rate']:  # 新变量统计量，下同
            df['X{}_{}_mean'.format(i, t)] = df[['X{}_{}_43'.format(i, t), 'X{}_{}_32'.format(i, t), 'X{}_{}_21'.format(i, t)]].mean(axis=1)
            df['X{}_{}_std'.format(i, t)] = df[['X{}_{}_43'.format(i, t), 'X{}_{}_32'.format(i, t), 'X{}_{}_21'.format(i, t)]].std(axis=1)
            df['X{}_{}_sum'.format(i, t)] = df[['X{}_{}_43'.format(i, t), 'X{}_{}_32'.format(i, t), 'X{}_{}_21'.format(i, t)]].sum(axis=1)
            df['X{}_{}_max'.format(i, t)] = df[['X{}_{}_43'.format(i, t), 'X{}_{}_32'.format(i, t), 'X{}_{}_21'.format(i, t)]].max(axis=1)
            df['X{}_{}_min'.format(i, t)] = df[['X{}_{}_43'.format(i, t), 'X{}_{}_32'.format(i, t), 'X{}_{}_21'.format(i, t)]].min(axis=1)

    collist = ['sum_X1238', 'sum_X456', 'sum_X1238_div_sum_X456', 'sum_all', 'X7_div_sum_X1238', 'X7_sub_sum_X1238',
               'X7_div_sum_X456', 'X7_sub_sum_X456', 'X7_div_sum_all', 'X7_sub_sum_all']
    for c in collist:
        for t in ['sub', 'rate']:
            df['{}_{}_mean'.format(c, t)] = df[['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].mean(axis=1)
            df['{}_{}_std'.format(c, t)] = df[['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].std(axis=1)
            df['{}_{}_sum'.format(c, t)] = df[['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].sum(axis=1)
            df['{}_{}_max'.format(c, t)] = df[['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].max(axis=1)
            df['{}_{}_min'.format(c, t)] = df[['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].min(axis=1)
    return df


def aum_feat_engineering(df):
    # 单个变量构造4个月数据交叉做差和做差商
    for i in range(1, 9):
        for j in range(4, 0, -1):
            for k in range(j-1, 0, -1):
                df['X{}_sub_{}{}'.format(i, j, k)] = df['X{}_{}'.format(i, j)] - df['X{}_{}'.format(i, k)]
                df['X{}_rate_{}{}'.format(i, j, k)] = (df['X{}_{}'.format(i, j)] - df['X{}_{}'.format(i, k)]) / (
                            1e-3 + df['X{}_{}'.format(i, k)])

    # 挖掘变量sun_X1238_{}（各式存款总额）
    for j in range(1, 5):
        df['sum_X1238_{}'.format(j)] = df[[
            'X1_{}'.format(j), 'X2_{}'.format(j), 'X3_{}'.format(j), 'X8_{}'.format(j)]].sum(axis=1)
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['sum_X1238_sub_{}{}'.format(j, k)] = df['sum_X1238_{}'.format(j)] - df['sum_X1238_{}'.format(k)]
            df['sum_X1238_rate_{}{}'.format(j, k)] = (df['sum_X1238_{}'.format(j)] - df['sum_X1238_{}'.format(k)]
                                                      ) / (1e-3 + df['sum_X1238_{}'.format(k)])

    # 挖掘变量sun_X456_{} （理财基金资管总额）
    for j in range(1, 5):
        df['sum_X456_{}'.format(j)] = df[['X4_{}'.format(j), 'X5_{}'.format(j), 'X6_{}'.format(j)]].sum(axis=1)
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['sum_X456_sub_{}{}'.format(j, k)] = df['sum_X456_{}'.format(j)] - df['sum_X456_{}'.format(k)]
            df['sum_X456_rate_{}{}'.format(j, k)] = (df['sum_X456_{}'.format(j)] - df['sum_X456_{}'.format(k)]
                                                      ) / (1e-3 + df['sum_X456_{}'.format(k)])

    # 挖掘变量sum_X1238_div_sum_X456_{} (上述两变量之比)
    for j in range(1, 5):
        df['sum_X1238_div_sum_X456_{}'.format(j)] = (df['sum_X1238_{}'.format(j)]) / (1e-3 + df['sum_X456_{}'.format(j)])
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['sum_X1238_div_sum_X456_sub_{}{}'.format(j, k)] = df['sum_X1238_div_sum_X456_{}'.format(j)
                                                                 ] - df['sum_X1238_div_sum_X456_{}'.format(k)]
            df['sum_X1238_div_sum_X456_rate_{}{}'.format(j, k)] = (df['sum_X1238_div_sum_X456_{}'.format(j)] - df[
                'sum_X1238_div_sum_X456_{}'.format(k)]) / (1e-3 + df['sum_X1238_div_sum_X456_{}'.format(k)])

    # 挖掘变量sun_all_{}
    for j in range(1, 5):
        df['sum_all_{}'.format(j)] = df['sum_X1238_{}'.format(j)] + df['sum_X456_{}'.format(j)]
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['sum_all_sub_{}{}'.format(j, k)] = df['sum_all_{}'.format(j)] - df['sum_all_{}'.format(k)]
            df['sum_all_rate_{}{}'.format(j, k)] = (df['sum_all_{}'.format(j)] - df['sum_all_{}'.format(k)]
                                                      ) / (1e-3 + df['sum_all_{}'.format(k)])

    # 构造上述总额与贷款之间关系变量
    for j in range(1, 5):
        df['X7_div_sum_X1238_{}'.format(j)] = df['X7_{}'.format(j)] / (1e-3 + df['sum_X1238_{}'.format(j)])
        df['X7_sub_sum_X1238_{}'.format(j)] = df['X7_{}'.format(j)] - df['sum_X1238_{}'.format(j)]

        df['X7_div_sum_X456_{}'.format(j)] = df['X7_{}'.format(j)] / (1e-3 + df['sum_X456_{}'.format(j)])
        df['X7_sub_sum_X456_{}'.format(j)] = df['X7_{}'.format(j)] - df['sum_X456_{}'.format(j)]

        df['X7_div_sum_all_{}'.format(j)] = df['X7_{}'.format(j)] / (1e-3 + df['sum_all_{}'.format(j)])
        df['X7_sub_sum_all_{}'.format(j)] = df['X7_{}'.format(j)] - df['sum_all_{}'.format(j)]

    df['X7_sub_sum_X1238_sum'] = df[['X7_sub_sum_X1238_2', 'X7_sub_sum_X1238_3', 'X7_sub_sum_X1238_4']].sum(axis=1)
    df['X7_sub_sum_X456_sum'] = df[['X7_sub_sum_X456_2', 'X7_sub_sum_X456_3', 'X7_sub_sum_X456_4']].sum(axis=1)
    df['X7_sub_sum_all_sum'] = df[['X7_sub_sum_all_2', 'X7_sub_sum_all_3', 'X7_sub_sum_all_4']].sum(axis=1)

    # 挖掘变量X7_div_sum_X1238_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_div_sum_X1238_sub_{}{}'.format(j, k)] = df['X7_div_sum_X1238_{}'.format(j)
                                                           ] - df['X7_div_sum_X1238_{}'.format(k)]
            df['X7_div_sum_X1238_rate_{}{}'.format(j, k)] = (df['X7_div_sum_X1238_{}'.format(j)] - df[
                'X7_div_sum_X1238_{}'.format(k)]) / (1e-3 + df['X7_div_sum_X1238_{}'.format(k)])

    # 挖掘变量X7_sub_sum_X1238_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_sub_sum_X1238_sub_{}{}'.format(j, k)] = df['X7_sub_sum_X1238_{}'.format(j)
                                                           ] - df['X7_sub_sum_X1238_{}'.format(k)]
            df['X7_sub_sum_X1238_rate_{}{}'.format(j, k)] = (df['X7_sub_sum_X1238_{}'.format(j)] - df[
                'X7_sub_sum_X1238_{}'.format(k)]) / (1e-3 + df['X7_sub_sum_X1238_{}'.format(k)])

    # 挖掘变量X7_div_sum_X456_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_div_sum_X456_sub_{}{}'.format(j, k)] = df['X7_div_sum_X456_{}'.format(j)
                                                           ] - df['X7_div_sum_X456_{}'.format(k)]
            df['X7_div_sum_X456_rate_{}{}'.format(j, k)] = (df['X7_div_sum_X456_{}'.format(j)] - df[
                'X7_div_sum_X456_{}'.format(k)]) / (1e-3 + df['X7_div_sum_X456_{}'.format(k)])

    # 挖掘变量X7_sub_sum_X456_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_sub_sum_X456_sub_{}{}'.format(j, k)] = df['X7_sub_sum_X456_{}'.format(j)
                                                           ] - df['X7_sub_sum_X456_{}'.format(k)]
            df['X7_sub_sum_X456_rate_{}{}'.format(j, k)] = (df['X7_sub_sum_X456_{}'.format(j)] - df[
                'X7_sub_sum_X456_{}'.format(k)]) / (1e-3 + df['X7_sub_sum_X456_{}'.format(k)])

    # 挖掘变量X7_div_sum_all_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_div_sum_all_sub_{}{}'.format(j, k)] = df['X7_div_sum_all_{}'.format(j)
                                                           ] - df['X7_div_sum_all_{}'.format(k)]
            df['X7_div_sum_all_rate_{}{}'.format(j, k)] = (df['X7_div_sum_all_{}'.format(j)] - df[
                'X7_div_sum_all_{}'.format(k)]) / (1e-3 + df['X7_div_sum_all_{}'.format(k)])

    # 挖掘变量X7_sub_sum_all_{}
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['X7_sub_sum_all_sub_{}{}'.format(j, k)] = df['X7_sub_sum_all_{}'.format(j)
                                                           ] - df['X7_sub_sum_all_{}'.format(k)]
            df['X7_sub_sum_all_rate_{}{}'.format(j, k)] = (df['X7_sub_sum_all_{}'.format(j)] - df[
                'X7_sub_sum_all_{}'.format(k)]) / (1e-3 + df['X7_sub_sum_all_{}'.format(k)])

    return df


def statistics_feature_cunkuan(df):
    for i in range(1, 4):
        df['C{}_mean'.format(i)] = df[['C{}_4'.format(i), 'C{}_3'.format(i), 'C{}_2'.format(i)]].mean(axis=1)
        df['C{}_std'.format(i)] = df[['C{}_4'.format(i), 'C{}_3'.format(i), 'C{}_2'.format(i)]].std(axis=1)
        df['C{}_sum'.format(i)] = df[['C{}_4'.format(i), 'C{}_3'.format(i), 'C{}_2'.format(i)]].sum(axis=1)
        df['C{}_max'.format(i)] = df[['C{}_4'.format(i), 'C{}_3'.format(i), 'C{}_2'.format(i)]].max(axis=1)
        df['C{}_min'.format(i)] = df[['C{}_4'.format(i), 'C{}_3'.format(i), 'C{}_2'.format(i)]].min(axis=1)

        for t in ['sub', 'rate']:
            df['C{}_{}_mean'.format(i, t)] = df[['C{}_{}_43'.format(i, t), 'C{}_{}_32'.format(i, t), 'C{}_{}_21'.format(i, t)]].mean(axis=1)
            df['C{}_{}_std'.format(i, t)] = df[['C{}_{}_43'.format(i, t), 'C{}_{}_32'.format(i, t), 'C{}_{}_21'.format(i, t)]].std(axis=1)
            df['C{}_{}_sum'.format(i, t)] = df[['C{}_{}_43'.format(i, t), 'C{}_{}_32'.format(i, t), 'C{}_{}_21'.format(i, t)]].sum(axis=1)
            df['C{}_{}_max'.format(i, t)] = df[['C{}_{}_43'.format(i, t), 'C{}_{}_32'.format(i, t), 'C{}_{}_21'.format(i, t)]].max(axis=1)
            df['C{}_{}_min'.format(i, t)] = df[['C{}_{}_43'.format(i, t), 'C{}_{}_32'.format(i, t), 'C{}_{}_21'.format(i, t)]].min(axis=1)

    return df


def cunkuan_feat_engineering(df):
    for i in range(1, 4):
        for j in range(4, 0, -1):
            for k in range(j-1, 0, -1):
                df['C{}_sub_{}{}'.format(i, j, k)] = df['C{}_{}'.format(i, j)] - df['C{}_{}'.format(i, k)]
                df['C{}_rate_{}{}'.format(i, j, k)] = (df['C{}_{}'.format(i, j)] - df['C{}_{}'.format(i, k)]) / (
                            1e-3 + df['C{}_{}'.format(i, k)])

    return df

def statistics_feature_beh(df):
    for i in range(1, 10):
        df['B{}_mean'.format(i)] = df[['B{}_4'.format(i), 'B{}_3'.format(i), 'B{}_2'.format(i)]].mean(axis=1)
        df['B{}_std'.format(i)] = df[['B{}_4'.format(i), 'B{}_3'.format(i), 'B{}_2'.format(i)]].std(axis=1)
        df['B{}_sum'.format(i)] = df[['B{}_4'.format(i), 'B{}_3'.format(i), 'B{}_2'.format(i)]].sum(axis=1)
        df['B{}_max'.format(i)] = df[['B{}_4'.format(i), 'B{}_3'.format(i), 'B{}_2'.format(i)]].max(axis=1)
        df['B{}_min'.format(i)] = df[['B{}_4'.format(i), 'B{}_3'.format(i), 'B{}_2'.format(i)]].min(axis=1)

        collist = ['B3_B5', 'B8_B9']
        for c in collist:
            for t in ['sub', 'rate']:
                df['{}_{}_mean'.format(c, t)] = df[
                    ['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].mean(axis=1)
                df['{}_{}_std'.format(c, t)] = df[
                    ['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].std(axis=1)
                df['{}_{}_sum'.format(c, t)] = df[
                    ['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].sum(axis=1)
                df['{}_{}_max'.format(c, t)] = df[
                    ['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].max(axis=1)
                df['{}_{}_min'.format(c, t)] = df[
                    ['{}_{}_43'.format(c, t), '{}_{}_32'.format(c, t), '{}_{}_21'.format(c, t)]].min(axis=1)
    return df


def beh_feat_engineering(df):
    for i in range(1, 10):
        for j in range(4, 0, -1):
            for k in range(j-1, 0, -1):
                df['B{}_sub_{}{}'.format(i, j, k)] = df['B{}_{}'.format(i, j)] - df['B{}_{}'.format(i, k)]
                df['B{}_rate_{}{}'.format(i, j, k)] = (df['B{}_{}'.format(i, j)] - df['B{}_{}'.format(i, k)]) / (
                            1e-3 + df['B{}_{}'.format(i, k)])

    #挖掘变量B3_B5_{} （当月转入减转出）
    for j in range(1, 5):
        df['B3_B5_{}'.format(j)] = df['B3_{}'.format(j)] - df['B5_{}'.format(j)]
    df['B3_B5_sum'] = df[['B3_B5_2', 'B3_B5_3', 'B3_B5_4']].sum(axis=1)  # 差值之和
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['B3_B5_sub_{}{}'.format(j, k)] = df['B3_B5_{}'.format(j)] - df['B3_B5_{}'.format(k)]
            df['B3_B5_rate_{}{}'.format(j, k)] = (df['B3_B5_{}'.format(j)] - df['B3_B5_{}'.format(k)]
                                                  ) / (1e-3 + df['B3_B5_{}'.format(k)])

    #挖掘变量B8_B9_{} （当月平均每次转入减转出）
    for j in range(1, 5):
        df['B8_B9_{}'.format(j)] = df['B8_{}'.format(j)] - df['B9_{}'.format(j)]
    df['B8_B9_sum'] = df[['B8_B9_2', 'B8_B9_3', 'B8_B9_4']].sum(axis=1)
    for j in range(4, 0, -1):
        for k in range(j-1, 0, -1):
            df['B8_B9_sub_{}{}'.format(j, k)] = df['B8_B9_{}'.format(j)] - df['B8_B9_{}'.format(k)]
            df['B8_B9_rate_{}{}'.format(j, k)] = (df['B8_B9_{}'.format(j)] - df['B8_B9_{}'.format(k)]
                                                  ) / (1e-3 + df['B8_B9_{}'.format(k)])

    # df['B6_less30'] = df['B6'].map(lambda x: 1 if x <= 31 else 0)
    return df


def event_feat_engineering(df):
    for i in range(1,19):
        df['E{}_less90'.format(i)] = df['E{}'.format(i)].map(lambda x: 1 if x <= 92 else 0)  # 该发生日期是否在当季度
        df['E{}_less30'.format(i)] = df['E{}'.format(i)].map(lambda x: 1 if x <= 31 else 0)  # 该发生日期是否在当月
    return df


def info_feat_engineering(df):
    # 年龄分层
    df["I2_18"] = df['I2'].map(lambda x: 1 if x >18 else 0)
    df["I2_25"] = df['I2'].map(lambda x: 1 if x >25 else 0)
    df["I2_30"] = df['I2'].map(lambda x: 1 if x >30 else 0)
    df["I2_40"] = df['I2'].map(lambda x: 1 if x >40 else 0)
    df["I2_50"] = df['I2'].map(lambda x: 1 if x >50 else 0)
    df["I2_60"] = df['I2'].map(lambda x: 1 if x >60 else 0)
    df["I2_70"] = df['I2'].map(lambda x: 1 if x >70 else 0)
    df["I2_80"] = df['I2'].map(lambda x: 1 if x >80 else 0)

    # 男女
    df['I1'] = df['I1'].map(lambda x: 1 if x == '男性' else 0)

    return df


def eval_error(pred, train_set):
    labels = train_set.get_label()
    pred = pred.reshape((3, int(len(pred) / 3))).T
    y_pred = pred.argmax(axis=1)
    score = cohen_kappa_score(labels, y_pred)
    return 'kappa_score', score, True


def model_train(df, trainlabel, cate_cols, test_, feature, num_class):
    '''
    @param df: 训练数据 DataFrame
    @param trainlabel：训练标签 string  eg. 'label'
    @param cate_cols: 类别变量名 list  eg. ['col1','col2'...]
    @param test_ : 测试数据 DataFrame
    @param feature ：所有训练特征 list  eg. ['feat1','feat2'...]

    @return sub_preds: 预测数据

    '''
    train_ = df.copy()
    n_splits = 5
    oof_lgb = np.zeros([len(train_), num_class])
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    # stratifiedKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    sub_preds = np.zeros([test_.shape[0], num_class])
    sub_preds1 = np.zeros([test_.shape[0], n_splits])
    use_cart = True
    cate_cols = cate_cols
    label = trainlabel
    pred = list(feature)
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        # 'metric':'multi-error',
        'num_class': num_class,
        'num_leaves': 60,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 5,
        'min_data_in_leaf': 20,
        'max_depth': -1,
        'nthread': 8,
        'verbose': 1,
        # 'is_unbalanace':True,
        # 'lambda_l1': 0.4,
        # 'lambda_l2': 0.5,
        # 'device': 'gpu'
    }

    valid_kappa = 0
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_[pred], train_[[label]]), start=1):
        print('the %s training start ...' % n_fold)

        train_x, train_y, train_weight = train_[pred].iloc[train_idx], train_[[label]].iloc[train_idx], \
                                         train_[['weight']].iloc[train_idx]
        valid_x, valid_y, valid_weight = train_[pred].iloc[valid_idx], train_[[label]].iloc[valid_idx], \
                                         train_[['weight']].iloc[valid_idx]

        print(train_y.shape)

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols,
                                 weight=train_weight.values.flatten(order='F'))
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)

        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=eval_error,
        )

        fold_importance_df = pd.DataFrame()
        fold_importance_df['Feature'] = feature
        fold_importance_df['importance'] = clf.feature_importance()
        fold_importance_df['fold'] = n_fold
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        sub_preds += clf.predict(test_[pred].values, num_iteration=clf.best_iteration) / folds.n_splits
        # sub_preds1[:, n_fold - 1] = clf.predict(test_[pred].values, num_iteration=400).argmax(axis=1)
        valid_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
        oof_lgb[valid_idx] = valid_pred
        y_pred = valid_pred.argmax(axis=1)
        fold_valid_kappa = cohen_kappa_score(valid_y, y_pred)
        valid_kappa += fold_valid_kappa

    # 计算N折平均Kappa值
    print('The average valid kappa_score:{:.6f}'.format(valid_kappa / n_splits))

    # plot feature importance
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
                                                                                                   ascending=False).index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',
                                                                                                    ascending=False)
    plt.figure(figsize=(8, 10))
    sns.barplot(y="Feature",
                x="importance",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()

    return sub_preds, oof_lgb, clf, sub_preds1, cols


def main():
    gen_pesudo_data()
    df_aum = get_data(data_file='aum', param='load_aum_data')
    df_aum_test = get_quater_data(df_aum, quater=5, colhead='X', num_col=8)
    df_aum_test_Q4 = get_quater_data(df_aum, quater=4, colhead='X', num_col=8)
    df_aum_test_Q3 = get_quater_data(df_aum, quater=3, colhead='X', num_col=8)

    df_aum_test = aum_feat_engineering(df_aum_test)
    df_aum_test_Q4 = aum_feat_engineering(df_aum_test_Q4)
    df_aum_test_Q3 = aum_feat_engineering(df_aum_test_Q3)
    df_aum_test = statistics_feature_aum(df_aum_test)
    df_aum_test_Q4 = statistics_feature_aum(df_aum_test_Q4)
    df_aum_test_Q3 = statistics_feature_aum(df_aum_test_Q3)


    df_cunkuan = get_data(data_file='cunkuan', param='load_cunkuan_data')
    df_cunkuan_test = get_quater_data(df_cunkuan, quater=5, colhead='C', num_col=3)
    df_cunkuan_test_Q4 = get_quater_data(df_cunkuan, quater=4, colhead='C', num_col=3)
    df_cunkuan_test_Q3 = get_quater_data(df_cunkuan, quater=3, colhead='C', num_col=3)

    df_cunkuan_test = cunkuan_feat_engineering(df_cunkuan_test)
    df_cunkuan_test_Q4 = cunkuan_feat_engineering(df_cunkuan_test_Q4)
    df_cunkuan_test_Q3 = cunkuan_feat_engineering(df_cunkuan_test_Q3)
    df_cunkuan_test = statistics_feature_cunkuan(df_cunkuan_test)
    df_cunkuan_test_Q4 = statistics_feature_cunkuan(df_cunkuan_test_Q4)
    df_cunkuan_test_Q3 = statistics_feature_cunkuan(df_cunkuan_test_Q3)


    df_beh = get_data(data_file='behavior', param='load_beh_data')
    df_beh_test = get_quater_data(df_beh, quater=5, colhead='B', num_col=9)
    df_beh_test_Q4 = get_quater_data(df_beh, quater=4, colhead='B', num_col=9)
    df_beh_test_Q3 = get_quater_data(df_beh, quater=3, colhead='B', num_col=9)

    df_beh_test = beh_feat_engineering(df_beh_test)
    df_beh_test_Q4 = beh_feat_engineering(df_beh_test_Q4)
    df_beh_test_Q3 = beh_feat_engineering(df_beh_test_Q3)
    df_beh_test = statistics_feature_beh(df_beh_test)
    df_beh_test_Q4 = statistics_feature_beh(df_beh_test_Q4)
    df_beh_test_Q3 = statistics_feature_beh(df_beh_test_Q3)


    df_event_test = get_data(data_file='big_event', param='load_big_event_data', Q=1)
    df_event_test_Q4 = get_data(data_file='big_event', param='load_big_event_data', Q=4)
    df_event_test_Q3 = get_data(data_file='big_event', param='load_big_event_data', Q=3)
    df_event_test = event_feat_engineering(df_event_test)
    df_event_test_Q4 = event_feat_engineering(df_event_test_Q4)
    df_event_test_Q3 = event_feat_engineering(df_event_test_Q3)


    df_info_test = get_data(data_file='info', param='load_info_data', Q=1)
    df_info_test_Q4 = get_data(data_file='info', param='load_info_data', Q=4)
    df_info_test_Q3 = get_data(data_file='info', param='load_info_data', Q=3)
    df_info_test = info_feat_engineering(df_info_test)
    df_info_test_Q4 = info_feat_engineering(df_info_test_Q4)
    df_info_test_Q3 = info_feat_engineering(df_info_test_Q3)


    cust_avli_Q3 = pd.read_csv('./train/avli/cust_avli_Q3.csv')
    cust_avli_Q4 = pd.read_csv('./train/avli/cust_avli_Q4.csv')
    cust_avli_test = pd.read_csv('./test/avli/cust_avli_Q1.csv')

    cust_avli_Q3 = df_aum_test_Q3.loc[df_aum_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)]
    cust_avli_Q4 = df_aum_test_Q4.loc[df_aum_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)]
    cust_avli_test = df_aum_test.loc[df_aum_test.cust_no.isin(cust_avli_test.cust_no.values)]

    cust_avli_Q3 = cust_avli_Q3.merge(
        df_cunkuan_test_Q3.loc[df_cunkuan_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)], on='cust_no', how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(
        df_cunkuan_test_Q4.loc[df_cunkuan_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)], on='cust_no', how='left')
    cust_avli_test = cust_avli_test.merge(
        df_cunkuan_test.loc[df_cunkuan_test.cust_no.isin(cust_avli_test.cust_no.values)], on='cust_no', how='left')

    cust_avli_Q3 = cust_avli_Q3.merge(
        df_beh_test_Q3.loc[df_beh_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)], on='cust_no', how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(
        df_beh_test_Q4.loc[df_beh_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)], on='cust_no', how='left')
    cust_avli_test = cust_avli_test.merge(
        df_beh_test.loc[df_beh_test.cust_no.isin(cust_avli_test.cust_no.values)], on='cust_no', how='left')

    cust_avli_Q3 = cust_avli_Q3.merge(
        df_event_test_Q3.loc[df_event_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)], on='cust_no', how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(
        df_event_test_Q4.loc[df_event_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)], on='cust_no', how='left')
    cust_avli_test = cust_avli_test.merge(
        df_event_test.loc[df_event_test.cust_no.isin(cust_avli_test.cust_no.values)], on='cust_no', how='left')

    cust_avli_Q3 = cust_avli_Q3.merge(
        df_info_test_Q3.loc[df_info_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)], on='cust_no', how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(
        df_info_test_Q4.loc[df_info_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)], on='cust_no', how='left')
    cust_avli_test = cust_avli_test.merge(
        df_info_test.loc[df_info_test.cust_no.isin(cust_avli_test.cust_no.values)], on='cust_no', how='left')


    label_Q3 = pd.read_csv('./train/y/y_Q3_3.csv')
    label_Q4 = pd.read_csv('./train/y/y_Q4_3.csv')
    label_Q3['label'] = label_Q3.label + 1
    label_Q4['label'] = label_Q4.label + 1
    cust_avli_Q3 = cust_avli_Q3.merge(label_Q3, on='cust_no', how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(label_Q4, on='cust_no', how='left')

    cust_avli_Q3['pre_label'] = np.nan
    last_label = label_Q3[['cust_no', 'label']]
    last_label.columns = ['cust_no', 'pre_label']
    cust_avli_Q4 = cust_avli_Q4.merge(last_label, on='cust_no', how='left')

    last_label = label_Q4[['cust_no', 'label']]
    last_label.columns = ['cust_no', 'pre_label']
    cust_avli_test = cust_avli_test.merge(last_label, on='cust_no', how='left')

    Train_data = pd.concat([cust_avli_Q3, cust_avli_Q4], axis=0)
    Train_data['weight'] = Train_data.label.map({1: 1.03, 2: 0.58, 0: 1})
    feature = Train_data.drop(['cust_no', 'label', 'weight'], axis=1).columns
    cate_cols = ['I3', 'I5', 'I8', 'I10', 'I12', 'I13', 'I14']
    for col in cate_cols:
        # Train_data[c] = Train_data[c].astype('category')
        # cust_avli_test[c] = cust_avli_test[c].astype('category')
        Train_data[col].fillna("0", inplace=True)
        cust_avli_test[col].fillna("0", inplace=True)
        le = LabelEncoder()
        le.fit(pd.concat([Train_data[col], cust_avli_test[col]], axis=0, ignore_index=True))
        Train_data[col] = le.transform(Train_data[col])
        cust_avli_test[col] = le.transform(cust_avli_test[col])

    Train_data.to_csv('./train_data.csv', index=None)
    cust_avli_test.to_csv('./test_data.csv', index=None)
    sub_preds, oof_lgb, clf, sub_preds1, cols = model_train(Train_data, trainlabel='label', cate_cols=cate_cols,
                                                      test_=cust_avli_test, feature=feature, num_class=3)

    # 保存特征重要性排序
    cols.to_list()
    c = np.array(cols)
    np.save('./test/importance_cols.npy', c)

    # 读取特征重要性
    cols = np.load('./test/importance_cols.npy', allow_pickle=True)
    cols.tolist()

    select_features = cols[0:300]
    cate_cols = [x for x in cate_cols if x in select_features]  # 求筛选的特征与cate特征的交集
    sub_preds, oof_lgb, clf, sub_preds1, cols1= model_train(Train_data, trainlabel='label', cate_cols=cate_cols,
                                                      test_=cust_avli_test, feature=select_features, num_class=3)

    cust_avli_test['label'] = sub_preds.argmax(axis=1) - 1
    cust_avli_test[['cust_no', 'label']].to_csv('./test/all_in_new_select300.csv', index=None)



if __name__ == "__main__":
    main()
