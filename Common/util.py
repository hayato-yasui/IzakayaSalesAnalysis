import os.path
import openpyxl as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Util:

    @staticmethod
    def create_dir(path):
        os.path.exists(os.mkdir(path))

    @staticmethod
    def check_existing_and_create_excel_file(file_path):
        if not os.path.exists(file_path):
            wb = px.Workbook()
            wb.save(file_path)

    @staticmethod
    def df_to_csv(df, dir, file_name, index=False):
        if not os.path.exists(dir):
            os.mkdir(dir)
        df.to_csv(dir + '/' + file_name, encoding='cp932', index=index)

    @staticmethod
    def moving_average(df, col_name, period):
        df['avg_' + col_name] = df[col_name].rolling(window=period).mean()
        return df

    @staticmethod
    def create_prd_and_obj_df_or_values(df, Y_col, df_or_values='df', does_replace_dummy=False):
        # X = Predictor variable , y = Objective variable
        X = df.drop(Y_col, axis=1)
        y = df[Y_col]
        if does_replace_dummy:
            X = pd.get_dummies(X, prefix='', prefix_sep='')
        if df_or_values == 'values':
            X = X.values
            y = y.values
        return X, y

    def leveling(self, df_src,sub_group_cols,main_group_cols,calc_tgt_cols,diff_tgt_col,diff_condition,does_output_csv=False,output_dir=None):
        df_calc_src, calc_tgt_dict = self._calc_tgt_sales(df_src,sub_group_cols,main_group_cols,calc_tgt_cols,diff_tgt_col,diff_condition)
        df_leveling_ratio = self._calc_sales_diff(df_calc_src, calc_tgt_cols, does_output_csv=does_output_csv,output_dir=output_dir)
        df_merged_ratio = pd.merge(df_src, df_leveling_ratio, on=main_group_cols)
        for c in calc_tgt_dict.keys():
            df_merged_ratio[c + '_平準化'] = df_merged_ratio.apply(
                lambda x: round(x[c] / x[c + '_増加率'],1) if x[diff_tgt_col] == diff_condition else x[c],axis=1)
        return df_merged_ratio

    @staticmethod
    def _calc_tgt_sales(df_src, sub_group_cols: list, main_group_cols: list, calc_tgt_cols: list, diff_tgt_col: str,
                        diff_condition):
        calc_tgt_dict = dict()
        for c in calc_tgt_cols:
            calc_tgt_dict.update({c: [c + '_sum', c + '_count']})
        df_grouped = df_src[sub_group_cols + [diff_tgt_col] + calc_tgt_cols]
        drop_cols = [c for c in sub_group_cols if c not in main_group_cols] + [diff_tgt_col]

        df_normal = df_grouped[df_grouped[diff_tgt_col] != diff_condition].groupby(
            sub_group_cols + [diff_tgt_col]).sum().reset_index().drop(drop_cols, axis=1)
        df_special = df_grouped[df_grouped[diff_tgt_col] == diff_condition].groupby(
            sub_group_cols + [diff_tgt_col]).sum().reset_index().drop(drop_cols, axis=1)
        dfs = [df_normal, df_special]
        for idx, df in enumerate(dfs):
            df = df.groupby(main_group_cols).agg(['sum', 'count']).reset_index()
            df.columns = ['_'.join(c) if c[1] != '' else c[0] for c in df.columns]
            for k, v_list in calc_tgt_dict.items():
                df[k + '_売上/日数'] = df[v_list[0]] / df[v_list[1]]
            dfs[idx] = df
        df_calc_src = pd.merge(dfs[0], dfs[1], on=main_group_cols, how='outer',
                               suffixes=('_normal', '_special')).set_index(main_group_cols)
        return df_calc_src, calc_tgt_dict

    def _calc_sales_diff(self, df_calc_src, calc_tgt_cols, does_output_csv=False,output_dir=None):
        return_cols = []
        for c in calc_tgt_cols:
            # nan -> 1
            df_calc_src[c + '_増加率'] = (df_calc_src[c + '_売上/日数_special'] / df_calc_src[c + '_売上/日数_normal']).replace(
                np.nan, 1)
            return_cols.append(c + '_増加率')
        if does_output_csv:
            index_names = '_'.join(df_calc_src.index.names)
            [self.df_to_csv(df_calc_src[c], output_dir, index_names + '_' + c + '.csv', True) for c in
             return_cols]
        return df_calc_src[return_cols].reset_index()
