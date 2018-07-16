import datetime
import numpy as np

TGT_PERIOD_FLOOR = datetime.date(2018, 4, 1)
TGT_PERIOD_TOP = datetime.date(2018, 6, 30)

# TGT_STORE = '大和乃山賊'
TGT_STORE = '定楽屋'
# TGT_STORE = 'うおにく'
# TGT_STORE = 'かこい屋'
# TGT_STORE = 'くつろぎ屋'
# TGT_STORE = 'ご馳走屋名駅店'
# TGT_STORE = 'ご馳走屋金山店'
# TGT_STORE = '九州乃山賊小倉総本店'
# TGT_STORE = '和古屋'
# TGT_STORE = '楽屋'
# TGT_STORE = '鳥Bouno!'
# TGT_STORE = 'ぐるめ屋'


class StoreCurrAnalysisSetting(object):
    TGT_STORE = TGT_STORE
    TGT_PERIOD_FLOOR = TGT_PERIOD_FLOOR
    TGT_PERIOD_TOP = TGT_PERIOD_TOP
    OUTPUT_DIR = './data/OUTPUT/' + TGT_STORE + '/'
    PIE_CHART_SET = ['商品カテゴリ2', '価格']

    GROUPING_WAY = {'価格': "sum"}

    TIME_SERIES_GRAPH_MONTHLY = ['売上', '来店総数']
    TIME_SERIES_GRAPH_DAYLY = ['売上', '来店総数']

    FIG_FILE_NAME = '定楽屋 金山店2018-04-01-2018-06-30.png'


class PreprocessSetting(object):
    TGT_STORE = TGT_STORE
    TGT_PERIOD_FLOOR = TGT_PERIOD_FLOOR
    TGT_PERIOD_TOP = TGT_PERIOD_TOP
    RAW_DATA_DIR = './data/Input/raw_data/'
    DATA_FILES_TO_FETCH = ['売上データ詳細_' + TGT_STORE + '_20180401-0630.csv', ]
    PROCESSED_DATA_DIR = './data/Input/processed_data/'+ TGT_STORE +'/'
    UNNECESSARY_COLS = ['親カテゴリ', 'H.合算先伝票番号', 'H.集計フラグ']
    DIVIDE_NECESSARY_COLS = ['D.商品']


    GROUPING_WAY = {'D.数量': "sum", 'D.価格': "sum", 'H.集計営業日': "min"}

    FILE_MEMO = '_before_grouping'

    TGT_TRANPOSE_C_AND_R_COL = ['D.商品名']
