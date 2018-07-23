import datetime
import numpy as np


class SrcConversion(object):
    REPLACE_UNEXPECTED_VAL_TO_ALT_VAL = {'D.オーダー日時': [['1203:アルバイト１', np.nan],
                                        ['1204:アルバイト２', np.nan]],'D.数量': [['0:設定なし', 0]], }
    REPLACE_NAN_TO_ALT_VAL = {'D.数量': 0, 'D.価格': 0, 'C.客層': '登録なし'}
    CONVERT_DTYPE = {'D.数量': 'numeric', 'D.価格': 'numeric','D.オーダー日時':'datetime','H.伝票発行日':'datetime',
                     'H.伝票処理日':'datetime','H.集計対象営業年月日':'datetime','H.伝票金額':'numeric',
                     'H.客数（合計）': 'numeric','H.客数（男）': 'numeric','H.客数（女）': 'numeric',}

    DIVIDE_NECESSARY_COLS = ['D.商品','H.店舗']
    UNNECESSARY_COLS_FOR_ALL_ANALYSIS = ['親カテゴリ', 'H.合算先伝票番号', 'H.集計フラグ']

class GroupingUnit(object):
    DAY_BILL_ORDER = ['H.集計対象営業年月日', 'H.伝票番号', 'D.オーダー日時']
    DAY_BILL = ['H.集計対象営業年月日', 'H.伝票番号']
    ITEM_CATEGORY2 = ['D.商品カテゴリ2']
    DOW = ['H.曜日', ]
    BILL = ['H.伝票番号',]

class MergeMasterData(object):
    # Store master
    F_PATH_STORE = './data/Input/master/store/store.csv'
    NECESSARY_COLS = ['店舗名','都道府県','営業開始時間','営業締め時間',]

    # weather master
    DIR_WEATHER = './data/Input/master/weather/'


