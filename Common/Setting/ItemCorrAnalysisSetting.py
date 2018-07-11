import datetime


class ItemCorrAnalysisSetting(object):
    CORR_LIMIT = 0.5


class PreprocessSetting(object):

    RAW_DATA_DIR = './data/Input/raw_data/'
    DATA_FILES_TO_FETCH = ['定楽屋 金山店2018-04-01-2018-06-30_before_grouping.csv', ]
    PROCESSED_DATA_DIR = './data/Input/processed_data/'
    UNNECESSARY_COLS = ['親カテゴリ', 'H.合算先伝票番号', 'H.集計フラグ']
    DIVIDE_NECESSARY_COLS = ['D.商品']

    # ToDo: format
    REPLACE_UNEXPECTED_VAL_TO_ALT_VAL = {'D.数量': ['0:設定なし', 0], }
    REPALCE_NAN_TO_ALT_VAL = {'D.数量': 0, 'D.価格': 0, 'D.商品カテゴリ3': 'dummy', }
    CONVERT_DTYPE = {'D.数量': float, 'D.価格': float}

    TGT_STORE = '定楽屋 金山店'
    TGT_PERIOD_FLOOR = datetime.date(2018, 4, 1)
    TGT_PERIOD_TOP = datetime.date(2018, 6, 30)

    GROUPING_KEY_DAY_BILL_ORDER = ['H.集計対象営業年月日', 'H.伝票番号', 'D.オーダー日時']

    GROUPING_WAY = {'D.数量': "sum", 'D.価格': "sum", 'H.集計営業日': "min"}

    GROUPING_FILE_MEMO = '縦横変換'

    TGT_TRANPOSE_C_AND_R_COL = ['D.商品名']
    TRANPOSE_C_AND_R_COUNT_COL = 'D.数量'
