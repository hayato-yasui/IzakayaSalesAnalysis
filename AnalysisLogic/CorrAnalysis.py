import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from Common.Logic.ChartClient import ChartClient
from Common.Logic.Preprocess import *
from Common.Logic.Postprocess import *
from Common.Setting.CorrAnalysisSetting import *
from Common.Setting.Common.PreprocessSetting import *
from Common.util import Util

# 商品間の相関係数算出クラス
class CorrAnalysis:

    def __init__(self):
        self.preproc_s = PreprocessSetting()
        self.ca_s = CorrAnalysisSetting()
        self.mmt = MergeMasterTable()
        self.mmt_s = MergeMasterTableSetting()
        self.preproc = Preprocess()
        self.postproc = Postprocess()
        self.chart_cli = ChartClient()
        self.gu = GroupingUnit()
        self.util = Util()

    def execute(self):
        self.df_preproc, preproc_csv_file_name = self._preprocess()
        # preproc_csv_file_name = ''
        # self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
        #                                                            , [preproc_csv_file_name])
        corr = self._calc_correlation(self.df_preproc)
        print(corr)
        # self._plot_corr(corr)

    def _preprocess(self):
        df_src = self.preproc.common_proc(self.preproc_s)
        df_item_pivot = self.preproc.tanspose_cols_and_rows(df_src, self.gu.DAY_BILL,
                                                            self.preproc_s.TGT_TRANPOSE_C_AND_R_COL,
                                                            self.preproc_s.TRANPOSE_C_AND_R_COUNT_COL)
        df_src = self.mmt.merge_store(df_src, self.mmt_s.F_PATH_STORE)
        df_src = self.mmt.merge_calender(df_src, self.mmt_s.F_PATH_CALENDER)
        df_src = self.mmt.merge_weather(df_src, self.mmt_s.DIR_WEATHER, self.preproc_s.TGT_PERIOD_FLOOR,
                                        self.preproc_s.TGT_PERIOD_TOP)

        self.preproc.convert_dtype_timedelta_to_int(df_src, '滞在時間')
        df_src['客構成'] = self.preproc.create_cstm_strctr(df_src)
        df_leveled = self.util.leveling(df_src, self.preproc_s.LEVELING_SUB_GROUP_COLS,
                                        self.preproc_s.LEVELING_MAIN_GROUP_COLS,
                                        self.preproc_s.LEVELING_CALC_TGT_COLS, self.preproc_s.LEVELING_DIFF_TGT_COL,
                                        self.preproc_s.LEVELING_DIFF_CONDITION, True, self.ca_s.OUTPUT_DIR)
        df_leveled['客単価/滞在時間'] = df_leveled['D.価格_平準化'] // df_leveled['滞在時間']
        # df_leveled['客単価高フラグ'] = df_leveled.apply(lambda x: 1 if x['客単価/滞在時間'] >= 4 else 0, axis=1)
        # df_grouped_by_bill = self.preproc.grouping(df_leveled, self.gu.DAY_BILL, self.preproc_s.GROUPING_WAY_BY_BILL)
        df_grouped_by_bill = df_leveled.groupby(self.gu.DAY_BILL).min().reset_index()


        df_grouped_by_bill = pd.merge(df_grouped_by_bill, df_item_pivot)
        # df_src = self.preproc.change_label_name(df_src)
        preproc_csv_file_name = self.preproc.create_proc_data_csv(df_grouped_by_bill, self.preproc_s.PROCESSED_DATA_DIR,
                                                                  self.preproc_s.TGT_STORE,
                                                                  self.preproc_s.TGT_PERIOD_FLOOR,
                                                                  self.preproc_s.TGT_PERIOD_TOP,
                                                                  '_' + self.preproc_s.GROUPING_FILE_MEMO)

        return df_grouped_by_bill, preproc_csv_file_name

    def _get_preproc_data(self, csv_file_name):
        return pd.read_csv(self.preproc_s.PROCESSED_DATA_DIR + csv_file_name, encoding='cp932')

    def _calc_correlation(self, df_preproc):
        df_corr = df_preproc.corr(method='pearson')['H.伝票金額_平準化']
        df_corr = df_corr[(df_corr>= self.ca_s.CORR_LIMIT) | (df_corr<= -self.ca_s.CORR_LIMIT)]
        print(1)
        return df_corr

    # def _plot_corr(self, df_corr):
    #     df_corr = df_corr[(df_corr >= self.ica_s.CORR_LIMIT) | (df_corr <= -self.ica_s.CORR_LIMIT)]
    #     df_corr.replace(1, np.nan, inplace=True)
    #     df_corr.dropna(how='all', inplace=True)
    #     df_corr.dropna(how='all', axis=1, inplace=True)
    #     sns.heatmap(df_corr, vmin=-1.0, vmax=1.0, center=0,
    #                 annot=True,  # True:格子の中に値を表示
    #                 fmt='.1f',
    #                 xticklabels=df_corr.columns.values,
    #                 yticklabels=df_corr.columns.values
    #                 )
    #     self.chart_cli.savefig(self.ica_s.OUTPUT_DIR,'商品間の相関係数.png')
    # self.chart_cli.plotfig()
    # self.chart_cli.closefig()


if __name__ == '__main__':
    ca = CorrAnalysis()
    ca.execute()
