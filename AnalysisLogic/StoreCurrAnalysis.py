import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Process

from Common.Logic.Preprocess import Preprocess
from Common.Logic.Postprocess import Postprocess
from Common.Logic.ChartClient import ChartClient
from Common.Setting.StoreCurrAnalysisSetting import *
from Common.Setting.Common.PreprocessSetting import *
from Common.util import Util

# 店舗の現状を把握する為に加工したデータをExcelや図に出力するクラス
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

    # def execute(self,tgt_store):
    def execute(self):
        # tgt_store = ['大和乃山賊', '定楽屋', 'うおにく', 'かこい屋', 'くつろぎ屋', 'ご馳走屋名駅店', 'ご馳走屋金山店',
        #              '九州乃山賊小倉総本店', '和古屋', '楽屋','鳥Bouno!', 'ぐるめ屋']
        tgt_store = ['大和乃山賊',]
        for s in tgt_store:
            self.sca_s.TGT_STORE = self.preproc_s.TGT_STORE = s
            self.sca_s.OUTPUT_DIR = './data/OUTPUT/' + self.sca_s.TGT_STORE + '/'
            self.preproc_s.DATA_FILES_TO_FETCH = ['売上データ詳細_' + self.preproc_s.TGT_STORE + '_20180401-0630.csv', ]
            self.preproc_s.PROCESSED_DATA_DIR = './data/Input/processed_data/' + self.preproc_s.TGT_STORE + '/'

            self.df_preproc,preproc_csv_file_name = self._preprocess()

            # preproc_csv_file_name = ''
            # self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
            #                                                            , [preproc_csv_file_name])
            # df_grouped_src = self.df_preproc.groupby(self.cols).mean().reset_index()
            # aaa = df_grouped_src[self.cols]
            # self.util.df_to_csv(aaa, self.sca_s.OUTPUT_DIR, '大和乃山賊＿サンプル.csv')

            self._output_store_curr_info()


            print(s + " is finish")

    def _preprocess(self):
        df_src = self.preproc.common_proc(self.preproc_s)

        self.preproc.dt_min_round(df_src,'注文時間',10)
        self.preproc.dt_min_round(df_src,'滞在時間',20)
        preproc_csv_file_name = self.preproc.create_proc_data_csv(df_src, self.preproc_s.PROCESSED_DATA_DIR,
                                                                  self.preproc_s.TGT_STORE,
                                                                  self.preproc_s.TGT_PERIOD_FLOOR,
                                                                  self.preproc_s.TGT_PERIOD_TOP,
                                                                  memo=self.preproc_s.FILE_MEMO)

        return df_src, preproc_csv_file_name


    def _output_store_curr_info(self):
        self.df_grouped_by_bill = self._create_df_grouped_by_bill()
        self.output_dict = dict()
        self._monthly_sales()
        self._daily_cstm_info()
        self._abc_analysis()
        [self._output_ord_tran(k_li) for k_li in self.sca_s.ORD_TRAN_KEY]

        with pd.ExcelWriter(self.sca_s.OUTPUT_DIR + self.sca_s.OUTPUT_F_EXCEL) as writer:
            [v_df.to_excel(writer, sheet_name=k) for k, v_df in self.output_dict.items()]

    def _create_df_grouped_by_bill(self):
        return self.df_preproc.groupby(self.gu.BILL).max().reset_index()

    def _output_ord_tran(self, key_li):
        df_grouped_by_order_time = self.df_preproc.groupby(key_li+['注文時間']).agg({'D.価格': np.sum, "D.数量": np.sum})
        self.output_dict.update({'注文推移_' + '_'.join(key_li): df_grouped_by_order_time})

    def _monthly_sales(self):
        self.df_grouped_by_month = self.df_grouped_by_bill.set_index(
            pd.DatetimeIndex(self.df_grouped_by_bill['H.集計対象営業年月日']))
        df_monthly_sales = self.df_grouped_by_month.groupby([
            self.df_grouped_by_month.index.year.rename('year'),
            self.df_grouped_by_month.index.month.rename('month')])['H.伝票金額'].sum()

        self.output_dict.update({'月間売上': df_monthly_sales})


    def _daily_cstm_info(self):
        df_daily_cstm = self.df_grouped_by_month.groupby(self.df_grouped_by_month.index).\
            agg(self.sca_s.GROUPING_WAY_DAILY_CSTM)
        self.output_dict.update({'日別客情報': df_daily_cstm})

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
        [self.sales_and_ratio_by_key(k, True) for k in self.sca_s.ABC_BILL_LEVEL_KEY]
        [self.sales_and_ratio_by_key(k, False) for k in self.sca_s.ABC_NO_BILL_LEVEL_KEY]

    def sales_and_ratio_by_key(self, key_li, bill_level_summary=True):
        df_grouped = self.df_preproc.groupby(key_li)
        df_sales_by_key = df_grouped.agg({'D.価格': np.sum,"H.客数（合計）": np.mean})
        # aaa = df_sales_by_key['D.価格'].sum()
        df_sales_by_key['売上比率'] = df_sales_by_key['D.価格'] / int(df_sales_by_key['D.価格'].sum())
        # df_sales_by_key['売上比率'] = df_sales_by_key / df_sales_by_key.sum()

        if bill_level_summary:
            df_tmp = self.df_preproc.groupby(self.gu.BILL + key_li).mean().reset_index().set_index(key_li, drop=True)
            s_count_by_key = df_tmp.groupby(key_li).size()
        else:
            s_count_by_key = df_grouped.size()
        s_count_by_key.name = 'count_' + '_'.join(key_li)
        s_ratio_by_key = pd.Series(s_count_by_key / s_count_by_key.sum(), name='ratio_' + '_'.join(key_li))

        df_merged = pd.concat([df_sales_by_key, s_count_by_key, s_ratio_by_key], axis=1)
        df_merged = self.preproc.sort_df(df_merged, ['count_' + '_'.join(key_li)], [False])
        df_merged["平均支払額"] = df_merged['D.価格'] / df_merged['count_' + '_'.join(key_li)]

        if key_li in self.sca_s.CALC_PRICE_PER_CSTM:
            df_merged["客単価"] = df_merged["平均支払額"] /df_merged["H.客数（合計）"]
        else:
            df_merged.drop(columns="H.客数（合計）",inplace=True)

        # self.chart_cli.create_pie_chart(df=df_merged, amount_col='D.価格')
        # self.chart_cli.savefig(self.sca_s.OUTPUT_DIR, 'ABC分析_売上構成比.png')

        self.output_dict.update({'ABC分析_' + '_'.join(key_li): df_merged})

        # self.util.df_to_csv(pd.concat([df_sales_by_key, s_count_by_key, s_ratio_by_key], axis=1),
        #                     self.sca_s.OUTPUT_DIR, 'ABC分析_' + '_'.join(key_li) + '.csv', index=True)

    # def sales_by_category2(self):
    #     df_grouped_by_category2 = self.preproc.grouping(self.df_preproc, self.gu.ITEM_CATEGORY2,
    #                                                     self.sca_s.GROUPING_WAY, self.sca_s.PIE_CHART_SET[0])
    #     df_grouped_by_category2 = self.preproc.sort_df(df_grouped_by_category2, ['D.価格'], [False])
    #     # self.chart_cli.create_pie_chart(df_grouped_by_category2,amount_col=self.sca_s.PIE_CHART_SET[1])
    #     df_grouped_by_category2['売上比率'] = df_grouped_by_category2 / df_grouped_by_category2.sum()
    #     df_grouped_by_category2.to_csv(self.sca_s.OUTPUT_DIR + 'ABC分析_売上構成比.csv')
    #     self.chart_cli.create_pie_chart(
    #         df=df_grouped_by_category2, amount_col=self.sca_s.PIE_CHART_SET[1])
    #     self.chart_cli.savefig(self.sca_s.OUTPUT_DIR, 'ABC分析_売上構成比.png')
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
    # store_li = ['大和乃山賊', '定楽屋', 'うおにく', 'かこい屋', 'くつろぎ屋', 'ご馳走屋名駅店', 'ご馳走屋金山店',
    #             '九州乃山賊小倉総本店', '和古屋', '楽屋', '鳥Bouno!', 'ぐるめ屋']
    # # Slice pro_plan base on process number
    # sls = len(store_li) // 3
    # store_li_sls = [store_li[sls * i:sls * (i + 1):] for i in range(3)]
    # proc = []
    # for ss in store_li_sls:
    #     proc.append(Process(target=sca.execute, args=(ss,)))
    # for p in proc:
    #     p.start()
    # for p in proc:
    #     p.join()
    sca.execute()
    print("END")