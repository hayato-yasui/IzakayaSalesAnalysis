import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from Common.Logic.Preprocess import Preprocess
from Common.Logic.Postprocess import Postprocess
from Common.Logic.ChartClient import ChartClient
from Common.Setting.StoreCurrAnalysisSetting import *


# 店舗の現状を把握する為のクラス
class StoreCurrAnalysis:

    def __init__(self):
        self.chart_cli = ChartClient()
        self.preproc_s = PreprocessSetting()
        self.sca_s = StoreCurrAnalysisSetting()
        self.preproc = Preprocess()

    def execute(self):
        preproc_csv_file_name = self._preprocess()
        # preproc_csv_file_name = ''
        self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
                                                                   , [preproc_csv_file_name])
        self._plot_store_curr_info()

    def _preprocess(self):
        df_src = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.RAW_DATA_DIR,
                                                          self.preproc_s.DATA_FILES_TO_FETCH)
        aaa = self.preproc.create_ord_time_col_from_enter_store(df_src)
        print(aaa['注文時間'])

        # self.preproc.del_unnecessary_cols(df_src, self.preproc_s.UNNECESSARY_COLS)
        # df_src = self.preproc.replace_values(df_src, self.preproc_s.REPLACE_UNEXPECTED_VAL_TO_ALT_VAL,
        #                                      self.preproc_s.REPALCE_NAN_TO_ALT_VAL)
        # df_src = self.preproc.divide_col(df_src, self.preproc_s.DIVIDE_NECESSARY_COLS)
        # df_src = self.preproc.convert_dtype(df_src, self.preproc_s.CONVERT_DTYPE)
        # df_src = self.preproc.deal_missing_values(df_src)
        # df_src = self.preproc.extract_data(df_src, self.preproc_s.TGT_STORE, self.preproc_s.TGT_PERIOD_FLOOR,
        #                                    self.preproc_s.TGT_PERIOD_TOP)
        # df_src = self.preproc.change_label_name(df_src)
        # preproc_csv_file_name = self.preproc.create_proc_data_csv(df_src, self.preproc_s.PROCESSED_DATA_DIR,
        #                                                           self.preproc_s.TGT_STORE,
        #                                                           self.preproc_s.TGT_PERIOD_FLOOR,
        #                                                           self.preproc_s.TGT_PERIOD_TOP,
        #                                                           memo=self.preproc_s.FILE_MEMO)

        return preproc_csv_file_name

    def _plot_store_curr_info(self):
        self._abc_analysis()

        # self.chart_cli.create_pie_chart(
        #     df=self.preproc.grouping(self.df_preproc, self.sca_s.GROUPING_KEY_ITEM_CATEGORY2,
        #                              self.sca_s.GROUPING_WAY, self.sca_s.PIE_CHART_SET[0]),
        #     amount_col=self.sca_s.PIE_CHART_SET[1])

        # 時系列カラムをインデックスに指定する必要がある
        # self.chart_cli.time_series_graph(self.df_preproc,
        #                                  amount_cols_li=self.df_preproc[self.sca_s.TIME_SERIES_GRAPH_MONTHLY])
        # self.chart_cli.time_series_graph(self.df_preproc,
        #                                  amount_cols_li=self.df_preproc[self.sca_s.TIME_SERIES_GRAPH_DAYLY])
        #

        # self.chart_cli.plotfig()
        # self.chart_cli.savefig(self.sca_s.OUTPUT_DIR + self.sca_s.FIG_FILE_NAME)
        # self.chart_cli.closefig()

    def _abc_analysis(self):
        df_grouped_by_category2 = self.preproc.grouping(self.df_preproc, self.sca_s.GROUPING_KEY_ITEM_CATEGORY2,
                                                        self.sca_s.GROUPING_WAY, self.sca_s.PIE_CHART_SET[0])

        df_grouped_by_category2 = self.preproc.sort_df(df_grouped_by_category2,['価格'],[False])
        # self.chart_cli.create_pie_chart(df_grouped_by_category2,amount_col=self.sca_s.PIE_CHART_SET[1])
        df_grouped_by_category2['売上比率'] = df_grouped_by_category2 / df_grouped_by_category2.sum()
        print(df_grouped_by_category2)
        self.chart_cli.create_pie_chart(
            df=df_grouped_by_category2, amount_col=self.sca_s.PIE_CHART_SET[1])
        self.chart_cli.plotfig()

if __name__ == '__main__':
    sca = StoreCurrAnalysis()
    sca.execute()
