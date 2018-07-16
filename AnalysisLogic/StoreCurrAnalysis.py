import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from Common.Logic.Preprocess import Preprocess
from Common.Logic.Postprocess import Postprocess
from Common.Logic.ChartClient import ChartClient
from Common.Setting.StoreCurrAnalysisSetting import *
from Common.Setting.Common.PreprocessSetting import *
from Common.util import Util


# 店舗の現状を把握する為のクラス
class StoreCurrAnalysis:
    # cols = ['H.伝票番号', 'H.集計対象営業年月日', 'H.曜日', 'H.伝票発行日', 'H.伝票処理日', 'H.テーブル番号', 'H.客数（合計）',
    #         'H.客数（男）', 'H.客数（女）', 'H.総商品数', 'H.伝票金額', 'H.伝票税額', 'C.客層']

    def __init__(self):
        self.chart_cli = ChartClient()
        self.preproc_s = PreprocessSetting()
        self.sca_s = StoreCurrAnalysisSetting()
        self.preproc = Preprocess()
        self.sc = SrcConversion()
        self.gu = GroupingUnit()
        self.util = Util()

    def execute(self):
        preproc_csv_file_name = self._preprocess()
        # preproc_csv_file_name = ''
        self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
                                                                   , [preproc_csv_file_name])
        # df_grouped_src = self.df_preproc.groupby(self.cols).mean().reset_index()
        # aaa = df_grouped_src[self.cols]
        # self.util.df_to_csv(aaa, self.sca_s.OUTPUT_DIR, '大和乃山賊＿サンプル.csv')
        self._plot_store_curr_info()

    def _preprocess(self):
        df_src = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.RAW_DATA_DIR,
                                                          self.preproc_s.DATA_FILES_TO_FETCH)

        self.preproc.del_unnecessary_cols(df_src, self.preproc_s.UNNECESSARY_COLS)
        # df_src = self.preproc.replace_values(df_src, self.sc.REPLACE_UNEXPECTED_VAL_TO_ALT_VAL,
        #                                      self.sc.REPALCE_NAN_TO_ALT_VAL)
        df_src = self.preproc.divide_col(df_src, self.preproc_s.DIVIDE_NECESSARY_COLS)
        df_src = self.preproc.convert_dtype(df_src, self.sc.CONVERT_DTYPE)
        # df_src = self.preproc.deal_missing_values(df_src)
        # df_src = self.preproc.extract_data(df_src, self.preproc_s.TGT_STORE, self.preproc_s.TGT_PERIOD_FLOOR,
        #                                    self.preproc_s.TGT_PERIOD_TOP)
        df_src = self.preproc.create_col_from_src_2cols(df_src, 'D.オーダー日時', 'H.伝票発行日', '注文時間')
        df_src = self.preproc.create_col_from_src_2cols(df_src, 'H.伝票処理日', 'H.伝票発行日', '滞在時間')
        df_src['滞在時間'] = df_src['滞在時間'].dt.round('20min')
        df_src['客構成'] = "男-" + df_src['H.客数（男）'].astype(str) + '人,女-' + df_src['H.客数（女）'].astype(str) + '人'
        # df_src = self.preproc.change_label_name(df_src)
        preproc_csv_file_name = self.preproc.create_proc_data_csv(df_src, self.preproc_s.PROCESSED_DATA_DIR,
                                                                  self.preproc_s.TGT_STORE,
                                                                  self.preproc_s.TGT_PERIOD_FLOOR,
                                                                  self.preproc_s.TGT_PERIOD_TOP,
                                                                  memo=self.preproc_s.FILE_MEMO)

        return preproc_csv_file_name

    def _plot_store_curr_info(self):
        self.sales()
        self._abc_analysis()

    def sales(self):
        self.df_grouped_by_bill = self.df_preproc.groupby(self.gu.BILL).max().reset_index()
        self.df_grouped_by_month = self.df_grouped_by_bill[['H.集計対象営業年月日','H.伝票金額']]\
            .groupby(pd.Grouper(key='H.集計対象営業年月日',freq='M')).size()

        # self.df_grouped_by_month = self.df_grouped_by_bill.groupby(pd.Grouper(freq='M')).sum()
        self.df_grouped_by_month = self.df_grouped_by_month + self.df_grouped_by_month.sum()
        pd.concat([self.df_grouped_by_month, self.df_grouped_by_month.sum()])

        print(self.df_grouped_by_month)

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
        self.sales_by_category2()
        self.customer()
        self.residence_time()


    def sales_by_category2(self):
        df_grouped_by_category2 = self.preproc.grouping(self.df_preproc, self.gu.ITEM_CATEGORY2,
                                                        self.sca_s.GROUPING_WAY, self.sca_s.PIE_CHART_SET[0])
        df_grouped_by_category2 = self.preproc.sort_df(df_grouped_by_category2, ['D.価格'], [False])
        # self.chart_cli.create_pie_chart(df_grouped_by_category2,amount_col=self.sca_s.PIE_CHART_SET[1])
        df_grouped_by_category2['売上比率'] = df_grouped_by_category2 / df_grouped_by_category2.sum()
        df_grouped_by_category2.to_csv(self.sca_s.OUTPUT_DIR + 'ABC分析_売上構成比.csv')
        self.chart_cli.create_pie_chart(
            df=df_grouped_by_category2, amount_col=self.sca_s.PIE_CHART_SET[1])
        self.chart_cli.savefig(self.sca_s.OUTPUT_DIR, 'ABC分析_売上構成比.png')
        # self.chart_cli.plotfig()

    def customer(self):
        self.df_grouped_by_bill = self.df_grouped_by_bill[['客構成']].reset_index(drop=True)
        self.df_grouped_by_customer = self.df_grouped_by_bill.groupby(['客構成']).size()
        print(self.df_grouped_by_customer)

    def residence_time(self):
        self.df_grouped_by_bill = self.df_grouped_by_bill[['滞在時間']].reset_index(drop=True)
        self.df_grouped_by_residence_time = self.df_grouped_by_bill.groupby(['滞在時間']).size()
        print(self.df_grouped_by_residence_time)




if __name__ == '__main__':
    sca = StoreCurrAnalysis()
    sca.execute()
