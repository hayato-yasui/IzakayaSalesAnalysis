import datetime
import numpy as np

class StoreCurrAnalysisSetting(object):
    PIE_CHART_SET = ['商品カテゴリ2','価格']

    GROUPING_KEY_ITEM_CATEGORY2 = ['商品カテゴリ2']
    GROUPING_WAY = {'価格': "sum"}

    TIME_SERIES_GRAPH_MONTHLY = ['売上','来店総数']
    TIME_SERIES_GRAPH_DAYLY = ['売上','来店総数']

    OUTPUT_DIR = './data/OUTPUT/'
    FIG_FILE_NAME = '定楽屋 金山店2018-04-01-2018-06-30.png'

class PreprocessSetting(object):
    RAW_DATA_DIR = './data/Input/raw_data/'
    DATA_FILES_TO_FETCH = ['売上データ詳細_定楽屋_20180401-0630.csv', ]
    PROCESSED_DATA_DIR = './data/Input/processed_data/'
    UNNECESSARY_COLS = ['親カテゴリ', 'H.合算先伝票番号', 'H.集計フラグ']
    DIVIDE_NECESSARY_COLS = ['D.商品']

    REPLACE_UNEXPECTED_VAL_TO_ALT_VAL = {'D.数量': ['0:設定なし', 0], 'D.オーダー日時':['1203:アルバイト１',np.nan]}
    REPALCE_NAN_TO_ALT_VAL = {'D.数量': 0, 'D.価格': 0, 'D.商品カテゴリ3': 'dummy', }
    CONVERT_DTYPE = {'D.数量': float, 'D.価格': float}

    TGT_STORE = '定楽屋 金山店'
    TGT_PERIOD_FLOOR = datetime.date(2018, 4, 1)
    TGT_PERIOD_TOP = datetime.date(2018, 6, 30)

    GROUPING_KEY_DAY_BILL_ORDER = ['H.集計対象営業年月日', 'H.伝票番号', 'D.オーダー日時']

    GROUPING_WAY = {'D.数量': "sum", 'D.価格': "sum", 'H.集計営業日': "min"}

    FILE_MEMO = '_before_grouping'

    TGT_TRANPOSE_C_AND_R_COL = ['D.商品名']
    TMP_COLS = GROUPING_KEY_DAY_BILL_ORDER + TGT_TRANPOSE_C_AND_R_COL

