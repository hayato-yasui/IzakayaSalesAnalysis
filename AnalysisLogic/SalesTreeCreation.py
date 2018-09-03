#-*- coding: utf-8 -*-
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdp
from sklearn import tree
from sklearn.externals.six import StringIO

from Common.Logic.ChartClient import ChartClient
from Common.Logic.Preprocess import *
from Common.Logic.Postprocess import *
from Common.Setting.SalesTreeCreationSetting import *
from Common.Setting.Common.PreprocessSetting import *
from Common.util import Util


class SalesTreeCreation:
    clf = DecisionTreeClassifier(random_state=0,max_depth=5)
    # clf = RandomForestClassifier(n_estimators=100,max_depth=5)

    def __init__(self):
        self.preproc_s = PreprocessSetting()
        self.stc_s = SalesTreeCreationSetting()
        self.preproc = Preprocess()
        self.postproc = Postprocess()
        self.mmt = MergeMasterTable()
        self.mmt_s = MergeMasterTableSetting()
        self.chart_cli = ChartClient()
        self.gu = GroupingUnit()
        self.util = Util()

    def execute(self):
        self.df_preproc, preproc_csv_file_name = self._preprocess()
        self._create_decision_tree()
        # preproc_csv_file_name = ''
        # self.df_preproc = self.preproc.fetch_csv_and_create_src_df(self.preproc_s.PROCESSED_DATA_DIR
        #                                                            , [preproc_csv_file_name])
        # self._postprocess()

    def _preprocess(self):
        df_src = self.preproc.common_proc(self.preproc_s)
        df_item_pivot = self.preproc.tanspose_cols_and_rows(df_src, self.gu.DAY_BILL,
                                                            self.preproc_s.TGT_TRANPOSE_C_AND_R_COL,
                                                            self.preproc_s.TRANPOSE_C_AND_R_COUNT_COL)
        df_src = self.mmt.merge_store(df_src, self.mmt_s.F_PATH_STORE)
        df_src = self.mmt.merge_weather(df_src, self.mmt_s.DIR_WEATHER, self.preproc_s.TGT_PERIOD_FLOOR,
                                        self.preproc_s.TGT_PERIOD_TOP)
        df_src = self.mmt.merge_calender(df_src, self.mmt_s.F_PATH_CALENDER)

        self.preproc.convert_dtype_timedelta_to_int(df_src, '滞在時間')
        df_src['客構成'] = self.preproc.create_cstm_strctr(df_src)
        df_leveled = self.util.leveling(df_src, self.preproc_s.LEVELING_SUB_GROUP_COLS,
                                        self.preproc_s.LEVELING_MAIN_GROUP_COLS,
                                        self.preproc_s.LEVELING_CALC_TGT_COLS, self.preproc_s.LEVELING_DIFF_TGT_COL,
                                        self.preproc_s.LEVELING_DIFF_CONDITION, True, self.stc_s.OUTPUT_DIR)
        df_leveled['客単価/滞在時間'] = df_leveled['D.価格_平準化'] // df_leveled['滞在時間']
        df_leveled['客単価高フラグ'] = df_leveled.apply(lambda x: 1 if x['客単価/滞在時間'] >= 5 else 0, axis=1)
        df_grouped_by_bill = self.preproc.grouping(df_leveled, self.gu.DAY_BILL, self.preproc_s.GROUPING_WAY_BY_BILL)
        # df_grouped_by_bill = pd.merge(df_grouped_by_bill, df_item_pivot)

        # preproc_csv_file_name = self.preproc.create_proc_data_csv(df_grouped_by_bill, self.preproc_s.PROCESSED_DATA_DIR,
        #                                                           self.preproc_s.TGT_STORE,
        #                                                           self.preproc_s.TGT_PERIOD_FLOOR,
        #                                                           self.preproc_s.TGT_PERIOD_TOP,
        #                                                           '_' + self.preproc_s.GROUPING_FILE_MEMO)

        # return df_grouped_by_bill, preproc_csv_file_name
        return df_grouped_by_bill, None

    def _get_preproc_data(self, csv_file_name):
        return pd.read_csv(self.preproc_s.PROCESSED_DATA_DIR + csv_file_name, encoding='cp932')

    def _create_decision_tree(self):
        self.df_preproc.drop(columns=['H.集計対象営業年月日', 'H.伝票番号'], inplace=True)
        self.preproc.replace_missing_value(self.df_preproc)
        X, y = self.util.create_prd_and_obj_df_or_values(self.df_preproc, '客単価高フラグ',does_replace_dummy=True)
        # df_train, df_test, label_train, label_test = train_test_split(X, y)

        self.clf.fit(X,y)
        predicted = self.clf.predict(X)
        print("========================================================")
        print("予測精度")
        print(sum(predicted == y) / len(y))

        dot_data = StringIO()
        tree.export_graphviz(self.clf,  # 決定木オブジェクトを一つ指定する
                                        out_file=dot_data,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                        filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                        rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                        feature_names=X.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                        # class_names=y.name,  # これを指定しないとチャート上で分類名が表示されない
                                        special_characters=True  # 特殊文字を扱えるようにする
                                        )
        graph = pdp.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(self.stc_s.OUTPUT_DIR + '決定木.pdf')

        # self.calc_elmnt_importance(X)

    def calc_elmnt_importance(self,X):
        # 特徴量の重要度
        feature = self.clf.feature_importances_

        # 特徴量の重要度を上から順に出力する
        f = pd.DataFrame({'number': range(0, len(feature)),
                          'feature': feature[:]})
        f2 = f.sort_values('feature', ascending=False)
        f3 = f2.ix[:, 'number']

        # 特徴量の名前
        label = X.columns[0:]

        # 特徴量の重要度順（降順）
        indices = np.argsort(feature)[::-1]

        for i in range(len(feature)):
            print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

        plt.title('Feature Importance')
        plt.bar(range(len(feature)), feature[indices], color='lightblue', align='center')
        plt.xticks(range(len(feature)), label[indices], rotation=90)
        plt.xlim([-1, len(feature)])
        plt.tight_layout()
        plt.show()

    def _standardization(self, X_train):
        scaler = StandardScaler()
        scaler.fit(X_train)


if __name__ == '__main__':
    stc = SalesTreeCreation()
    stc.execute()
