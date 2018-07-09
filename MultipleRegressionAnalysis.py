import os
import re
import pandas as pd
import numpy as np
import datetime
from enum import Enum, IntEnum
from sklearn import linear_model
import matplotlib.pyplot as plt

from Common.Logic.Preprocess import Preprocess
from Common.Logic.Postprocess import Postprocess
from Common.Setting.PreprocessSetting import PreprocessSetting
from Common.Setting.MultipleRegressionAnalysis import MultipleRegressionAnalysisSetting


# class PLAYER_ANSWER(Enum):
#     MIN = 'MIN'
#     MAX = 'MAX'
#     MID = 'MID'
#     PASS = 'PASS'


# 重回帰分析クラス
class MultipleRegressionAnalysis:
    clf = linear_model.LinearRegression()

    def __init__(self):
        self.preproc_s = PreprocessSetting()
        self.mra_s = MultipleRegressionAnalysisSetting()
        self.preproc = Preprocess()
        self.postproc = Postprocess()

    def execute(self):
        preproc_csv_path = self._preprocess()
        # preproc_csv_path = ''
        self.df_preproc = self._get_preproc_data(preproc_csv_path)
        corr = self._calc_correlation(self.df_preproc)
        print(corr)
        self._create_prediction_model()
        self._postprocess()

    def _preprocess(self):
        df_src = self.preproc.fetch_csv_data_and_convert_format_to_df(self.preproc_s.RAW_DATA_DIR,
                                                                      self.preproc_s.DATA_FILES_TO_FETCH)
        self.preproc.del_unnecessary_cols(df_src, self.preproc_s.UNNECESSARY_COLS)
        df_src = self.preproc.replace_values(df_src, self.preproc_s.REPLACE_UNEXPECTED_VAL_TO_ALT_VAL,
                                             self.preproc_s.REPALCE_NAN_TO_ALT_VAL)
        df_src = self.preproc.divide_col(df_src, self.preproc_s.DIVIDE_NECESSARY_COLS)
        df_src = self.preproc.convert_dtype(df_src, self.preproc_s.CONVERT_DTYPE)
        df_src = self.preproc.deal_missing_values(df_src)
        df_src = self.preproc.extract_data(df_src, self.preproc_s.TGT_STORE, self.preproc_s.TGT_PERIOD_FLOOR,
                                           self.preproc_s.TGT_PERIOD_TOP)
        preproc_csv_file_name = self.preproc.create_proc_data_csv(df_src, self.preproc_s.PROCESSED_DATA_DIR,
                                                                  self.preproc_s.TGT_STORE,
                                                                  self.preproc_s.TGT_PERIOD_FLOOR,
                                                                  self.preproc_s.TGT_PERIOD_TOP, '_before_grouping')

        # Do grouping though, no change df name due to being able to skip those process
        df_src = self.preproc.grouping(df_src, self.preproc_s.GROUPING_KEY_DOW, self.preproc_s.GROUPING_WAY)
        df_src = self.preproc.change_label_name(df_src)
        preproc_csv_file_name = self.preproc.create_proc_data_csv(df_src, self.preproc_s.PROCESSED_DATA_DIR,
                                                                  self.preproc_s.TGT_STORE,
                                                                  self.preproc_s.TGT_PERIOD_FLOOR,
                                                                  self.preproc_s.TGT_PERIOD_TOP,
                                                                  '_' + self.preproc_s.GROUPING_FILE_MEMO)

        return preproc_csv_file_name

    def _get_preproc_data(self, csv_file_name):
        return pd.read_csv(self.preproc_s.PROCESSED_DATA_DIR + csv_file_name, encoding='cp932')

    # for debug
    def _calc_correlation(self, df_preproc):
        return df_preproc.corr(method='pearson')

    def _create_prediction_model(self):
        self.df_sales_high_corr = self._del_lower_corr_cols()
        self._normalization()
        X, Y = self._create_prd_and_obj_valiables()
        self._create_model(X, Y)

    def _del_lower_corr_cols(self):
        # get columns that have high corr
        high_corr_cols_li = self.df_preproc.corr()[(self.df_preproc.corr()['価格'] >= self.mra_s.CORR_LIMIT) |
                                                   (self.df_preproc.corr()[
                                                        '価格'] <= -self.mra_s.CORR_LIMIT)].index.values
        return self.df_preproc[high_corr_cols_li]

    def _normalization(self):
        # データフレームの各列を正規化
        self.df_sales_high_corr = self.df_sales_high_corr.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        self.df_sales_high_corr.head()

    def _create_prd_and_obj_valiables(self):
        # X = Predictor variable , Y = Objective variable
        # X = self.df_sales_high_corr.drop('売上', axis=1).as_matrix()
        # Y = self.df_sales_high_corr['売上'].as_matrix()
        X = self.df_sales_high_corr.drop('価格', axis=1).values
        Y = self.df_sales_high_corr['価格'].values
        return X, Y

    def _create_model(self, X, Y):
        self.clf.fit(X, Y)
        # 偏回帰係数
        # print(pd.DataFrame({"Name": self.df_sales_high_corr.drop('売上', axis=1).columns,
        #                     "Coefficients": self.clf.coef_}).sort_values(by='Coefficients'))
        print(pd.DataFrame({"Name": self.df_sales_high_corr.drop('価格', axis=1).columns,
                            "Coefficients": self.clf.coef_}).sort_values(by='Coefficients'))
        # 切片 (誤差)
        print(self.clf.intercept_)

        # 散布図
        plt.scatter(X, Y)

        # 回帰直線
        plt.plot(X, self.clf.predict(X))
        # plt.show()


if __name__ == '__main__':
    mra = MultipleRegressionAnalysis()
    mra.execute()
