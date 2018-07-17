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
        store_li = ['大和乃山賊', '定楽屋', 'うおにく', 'かこい屋', 'くつろぎ屋', 'ご馳走屋名駅店', 'ご馳走屋金山店',
                     '九州乃山賊小倉総本店', '和古屋', '楽屋','鳥Bouno!', 'ぐるめ屋']
        for s in store_li:
            self.sca_s.TGT_STORE = s
            preproc_csv_file_name = self._preprocess()
            # preproc_csv_file_name = ''
            self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
                                                                       , [preproc_csv_file_name])
            # df_grouped_src = self.df_preproc.groupby(self.cols).mean().reset_index()
            # aaa = df_grouped_src[self.cols]
            # self.util.df_to_csv(aaa, self.sca_s.OUTPUT_DIR, '大和乃山賊＿サンプル.csv')
            self._plot_store_curr_info()
            print(s + " is finish")

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
        df_src['客構成'] = "男 : " + df_src['H.客数（男）'].astype(str) + '人, 女 : ' + df_src['H.客数（女）'].astype(str) + '人'
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
        self.df_grouped_by_bill = self.df_grouped_by_bill.set_index(
            pd.DatetimeIndex(self.df_grouped_by_bill['H.集計対象営業年月日']))
        self.df_grouped_by_month = self.df_grouped_by_bill.groupby([
            self.df_grouped_by_bill.index.year, self.df_grouped_by_bill.index.month])['H.伝票金額'].sum()
        total_sales = self.df_grouped_by_month.sum()
        self.util.df_to_csv(self.df_grouped_by_month,self.sca_s.OUTPUT_DIR,'月間売上.csv',index=True)

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
        self.sales_and_ratio_by_key(["客構成"])
        self.sales_and_ratio_by_key(["滞在時間"])

    def sales_and_ratio_by_key(self, key_li):
        df_grouped = self.df_grouped_by_bill.groupby(key_li)
        df_sales_by_key = df_grouped.agg({'D.価格': np.sum})
        df_sales_by_key['売上比率'] = df_sales_by_key / df_sales_by_key.sum()
        s_count_by_key = df_grouped.size()
        s_count_by_key.name = 'count_' + '_'.join(key_li)
        s_ratio_by_key = pd.Series(s_count_by_key / s_count_by_key.sum(), name='ratio_' + '_'.join(key_li))
        self.util.df_to_csv(pd.concat([df_sales_by_key, s_count_by_key, s_ratio_by_key], axis=1),
                            self.sca_s.OUTPUT_DIR, 'ABC分析_' + '_'.join(key_li) + '.csv', index=True)


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

    # def customer(self):
    #     df_grouped_by_customer = self.df_grouped_by_bill.groupby(['客構成'])
    #     df_sales_by_customer = df_grouped_by_customer.agg({'D.価格': np.sum})
    #     df_sales_by_customer['売上比率'] = df_sales_by_customer / df_sales_by_customer.sum()
    #     s_count_by_customer = df_grouped_by_customer.size()
    #     s_count_by_customer.name = 'count_customer'
    #     s_ratio = pd.Series(s_count_by_customer / s_count_by_customer.sum(), name='ratio_customer')
    #     self.util.df_to_csv(pd.concat([df_sales_by_customer,s_count_by_customer, s_ratio], axis=1),
    #                         self.sca_s.OUTPUT_DIR, 'ABC分析_客構成比.csv', index=True)
    #
    # def residence_time(self):
    #     df_grouped_by_residence_time = self.df_grouped_by_bill.groupby(['滞在時間'])
    #     df_sales_by_residence_time = df_grouped_by_residence_time.agg({'D.価格': np.sum})
    #     df_sales_by_residence_time['売上比率'] = df_sales_by_residence_time / df_sales_by_residence_time.sum()
    #     s_count_by_residence_time = df_grouped_by_residence_time.size()
    #     s_count_by_residence_time.name = 'count_residence_time'
    #     s_ratio = pd.Series(s_count_by_residence_time / s_count_by_residence_time.sum(), name='ratio_residence_time')
    #     self.util.df_to_csv(pd.concat([s_count_by_residence_time, s_ratio, df_sales_by_residence_time], axis=1),
    #                         self.sca_s.OUTPUT_DIR, 'ABC分析_滞在時間成比.csv',
    #                         index=True)


if __name__ == '__main__':
    sca = StoreCurrAnalysis()
    sca.execute()
    print("END")