import matplotlib.pyplot as plt


class ChartClient:
    @staticmethod
    def savefig(file_path):
        plt.savefig(file_path)

    @staticmethod
    def closefig(close_type="all"):
        plt.close(close_type)

    @staticmethod
    def plotfig():
        plt.show()

    @staticmethod
    def df_plotfig(df,subplots=False):
        df.plot(subplots=subplots)

    @staticmethod
    def create_pie_chart(df, amount_col,sort_columns=False):
        df.plot(kind='pie', y=amount_col,sort_columns=sort_columns)
        # plt.title('円グラフ', size=16, fontproperties=fp)

    @staticmethod
    def time_series_graph(df,amount_cols_li, figsize=(16, 4), alpha = 0.5):
        # 時系列カラムをインデックスに指定する必要がある
        df.plot(y=amount_cols_li, figsize=figsize, alpha=alpha)
