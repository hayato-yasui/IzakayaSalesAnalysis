import datetime
from Common.Logic.Preprocess import *
TGT_PERIOD_FLOOR = datetime.date(2018, 4, 1)
TGT_PERIOD_TOP = datetime.date(2018, 6, 30)

TGT_STORE = '大和乃山賊'
# TGT_STORE = '定楽屋'
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


class CorrAnalysisSetting(object):
    TGT_STORE = TGT_STORE
    TGT_PERIOD_FLOOR = TGT_PERIOD_FLOOR
    TGT_PERIOD_TOP = TGT_PERIOD_TOP
    CORR_LIMIT = 0.7
    OUTPUT_DIR = './data/OUTPUT/' + TGT_STORE + '/'


class PreprocessSetting(object):
    TGT_STORE = TGT_STORE
    TGT_PERIOD_FLOOR = TGT_PERIOD_FLOOR
    TGT_PERIOD_TOP = TGT_PERIOD_TOP

    RAW_DATA_DIR = './data/Input/raw_data/'
    DATA_FILES_TO_FETCH = ['売上データ詳細_' + TGT_STORE + '_20180401-0630.csv', ]
    # DATA_FILES_TO_FETCH = ['定楽屋 金山店2018-04-01-2018-06-30_before_grouping.csv', ]
    PROCESSED_DATA_DIR = './data/Input/processed_data/'+ TGT_STORE +'/'


    GROUPING_WAY_BY_BILL = {'H.曜日': "min", 'H.伝票発行日': "min", 'H.客数（合計）': "min",'H.客数（男）': "min",
                            'H.客数（女）': "min",'H.伝票金額': "min", '滞在時間': "min",}

    GROUPING_FILE_MEMO = '縦横変換'

    TGT_TRANPOSE_C_AND_R_COL = ['D.商品カテゴリ2']
    TRANPOSE_C_AND_R_COUNT_COL = 'D.数量'

    LEVELING_DIFF_TGT_COL = '翌日が休日'
    LEVELING_DIFF_CONDITION = 1
    LEVELING_CALC_TGT_COLS = ['D.価格','D.数量','H.伝票金額']
    LEVELING_SUB_GROUP_COLS = GroupingUnit.DAY_ITEM_CATEGORY2
    LEVELING_MAIN_GROUP_COLS = GroupingUnit.ITEM_CATEGORY2
    # SUB_GROUP_COLS = GroupingUnit.STORE_DAY
    # MAIN_GROUP_COLS = GroupingUnit.STORE