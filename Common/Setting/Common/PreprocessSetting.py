import datetime
import numpy as np


class SrcConversion(object):
    # ToDo: format
    REPLACE_UNEXPECTED_VAL_TO_ALT_VAL = {'D.オーダー日時': [['1203:アルバイト１', np.nan],
                                        ['1204:アルバイト２', np.nan]],'D.数量': [['0:設定なし', 0]], }
    REPALCE_NAN_TO_ALT_VAL = {'D.数量': 0, 'D.価格': 0, 'D.商品カテゴリ3': 'dummy', 'C.客層': '登録なし'}
    CONVERT_DTYPE = {'D.数量': 'numeric', 'D.価格': 'numeric','D.オーダー日時':'datetime','H.伝票発行日':'datetime',
                     'H.伝票処理日':'datetime','H.集計対象営業年月日':'datetime','H.伝票金額':'numeric'}

class GroupingUnit(object):
    DAY_BILL_ORDER = ['H.集計対象営業年月日', 'H.伝票番号', 'D.オーダー日時']
    ITEM_CATEGORY2 = ['D.商品カテゴリ2']
    DOW = ['H.曜日', ]
    BILL = ['H.伝票番号',]


