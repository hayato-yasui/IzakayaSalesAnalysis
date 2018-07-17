import os
import re
import pandas as pd
import numpy as np
import datetime
from enum import Enum, IntEnum
from sklearn import linear_model
import matplotlib.pyplot as plt


class Preprocess:

    @staticmethod
    def fetch_csv_and_create_src_df(data_dir, file_names_li):
        for idx, f in enumerate(file_names_li):
            if idx == 0:
                df_src = pd.read_csv(data_dir + f, encoding='cp932', engine='python')
            else:
                df_src = pd.concat([df_src, pd.read_csv(data_dir + f, encoding='cp932', engine='python')])
        return df_src

    @staticmethod
    def del_unnecessary_cols(df, unnecessary_cols):
        df.drop(columns=unnecessary_cols, axis=1, inplace=True)

    @staticmethod
    def divide_col(df, divide_necessary_cols):
        # divide cols that has ID and name like 店舗ID:店舗名 -> 店舗ID,店舗名
        for c in divide_necessary_cols:
            df = pd.concat([df, df[c].str.split(':', expand=True)], axis=1).drop(c, axis=1)
            df.rename(columns={0: c + 'ID', 1: c + '名'}, inplace=True)
        return df

    @staticmethod
    def change_label_name(df):
        df.rename(columns=lambda s: s[2:] if s.count(".") else s, inplace=True)
        return df

    @staticmethod
    def deal_missing_values(df, method='interpolate'):
        if method == 'interpolate':
            df.interpolate(inplace=True)
        return df

    @staticmethod
    def extract_data(df, tgt_store, tgt_period_floor, tgt_period_top):
        # df = df[df['店舗名']==TGT_STORE]
        # df = df[tgt_period_floor <= df_['伝票処理日'] <= tgt_period_top]
        return df

    @staticmethod
    def create_proc_data_csv(df, proc_data_dir, tgt_store, tgt_period_floor, tgt_period_top, memo='',index=False):
        output_csv_file_name = tgt_store + str(tgt_period_floor) + '-' + str(tgt_period_top) + memo + '.csv'
        if not os.path.exists(proc_data_dir):
            os.mkdir(proc_data_dir)
        df.to_csv(proc_data_dir + output_csv_file_name, index=index, encoding='cp932')
        return output_csv_file_name

    @staticmethod
    def replace_values(df, unexpected_val_dict, nan_val_dict):
        for k, v_list in unexpected_val_dict.items():
            [df[k].replace(v[0], v[1], inplace=True) for v in v_list]
        [df[k].fillna(v, inplace=True) for k, v in nan_val_dict.items()]
        return df

    @staticmethod
    def convert_dtype(df, dict):
        for k, v in dict.items():
            if v == 'numeric':
                df[k] = pd.to_numeric(df[k], errors='coerce')
            elif v == 'datetime':
                df[k] = pd.to_datetime(df[k], errors='coerce')
            else:
                df[k] = df[k].astype(v)
        return df

    @staticmethod
    def grouping(df, key_li, grouping_item_and_way_dict, index_col=None):
        selected_cols = key_li + [k for k, v in grouping_item_and_way_dict.items()]
        df_selected = df[selected_cols]
        df_grouped_src = df_selected.groupby(key_li)
        df_grouped = df_grouped_src.agg(grouping_item_and_way_dict).reset_index()
        if index_col is not None:
            df_grouped = df_grouped.set_index(index_col)
        return df_grouped

    @staticmethod
    def tanspose_cols_and_rows(df, keys_li, tgt_cols_li, count_col):
        selected_cols = keys_li + tgt_cols_li + [count_col]
        df_selected = df[selected_cols]
        df_pivot = df_selected.pivot_table(index=keys_li, columns=tgt_cols_li, values='D.数量', aggfunc=sum). \
            fillna(0).astype("int").reset_index()
        return df_pivot

    @staticmethod
    def outlier_2s(df):
        for i in range(len(df.columns)):
            # 列を抽出する
            col = df.iloc[:, i]

            # 平均と標準偏差
            average = np.mean(col)
            sd = np.std(col)

            # 外れ値の基準点
            outlier_min = average - (sd) * 2
            outlier_max = average + (sd) * 2

            # 範囲から外れている値を除く
            col[col < outlier_min] = None
            col[col > outlier_max] = None

        df.dropna(how='any', axis=0, inplace=True)
        return df

    @staticmethod
    def outlier_iqr(df):

        for i in range(len(df.columns)):
            # 列を抽出する
            col = df.iloc[:, i]

            # 四分位数
            q1 = col.describe()['25%']
            q3 = col.describe()['75%']
            iqr = q3 - q1  # 四分位範囲

            # 外れ値の基準点
            outlier_min = q1 - (iqr) * 1.5
            outlier_max = q3 + (iqr) * 1.5

            # 範囲から外れている値を除く
            col[col < outlier_min] = None
            col[col > outlier_max] = None

        df.dropna(how='any', axis=0, inplace=True)
        return df

    # sort_ways_li : ascending  -> True
    #                descending -> False
    @staticmethod
    def sort_df(df, sort_cols_li, sort_ways_li):
        return df.sort_values(sort_cols_li, ascending=sort_ways_li)

    @staticmethod
    def create_col_from_src_2cols(df, col1, col2, new_col, method='minus'):
        # method is selected in ('minus', 'plus', 'divide', 'times')
        df_tgt_cols = df[[col1, col2]]
        df_tgt_cols.dropna(how='any', inplace=True)
        if method == 'minus':
            df[new_col] = df_tgt_cols[col1] - df_tgt_cols[col2]
        elif method == 'plus':
            df[new_col] = df_tgt_cols[col1] + df_tgt_cols[col2]
        elif method == 'divide':
            df[new_col] = df_tgt_cols[col1] / df_tgt_cols[col2]
        elif method == 'plus':
            df[new_col] = df_tgt_cols[col1] * df_tgt_cols[col2]
        else:
            raise ValueError
        return df

    @staticmethod
    def convert_dtype_to_datetime(df, cols_li):
        [df[c].convert_objects().astype(np.datetime64) for c in cols_li]

    @staticmethod
    def convert_dtype_to_numeric(df, cols_li):
        [df[c].convert_objects(convert_numeric=True).astype(np.numeric) for c in cols_li]
