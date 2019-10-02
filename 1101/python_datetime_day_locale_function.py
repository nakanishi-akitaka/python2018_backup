# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-datetime-day-locale-function/
Created on Thu Nov  1 14:44:24 2018

@author: Akitaka
"""

import datetime
import locale

dt = datetime.datetime(2018, 1, 1)
print(dt)
# 2018-01-01 00:00:00

print(dt.strftime('%A, %a, %B, %b'))
# Monday, Mon, January, Jan

print(locale.getlocale(locale.LC_TIME))
# (None, None)

#%%
locale.setlocale(locale.LC_TIME, 'ja_JP.UTF-8')
print(locale.getlocale(locale.LC_TIME))
# ('ja_JP', 'UTF-8')

print(dt.strftime('%A, %a, %B, %b'))
# 月曜日, 月, 1月,  1

locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
print(dt.strftime('%A, %a, %B, %b'))
# Monday, Mon, January, Jan

locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
print(dt.strftime('%A, %a, %B, %b'))
# Montag, Mo, Januar, Jan

locale.setlocale(locale.LC_TIME, 'ja_JP.UTF-8')

s = '2018-01-01'
s_dow = datetime.datetime.strptime(s, '%Y-%m-%d').strftime('%A')

print(s_dow)
# 月曜日
#%%

import datetime

dt = datetime.datetime(2018, 1, 1)
print(dt)
# 2018-01-01 00:00:00

w_s = dt.strftime('%w')
print(w_s)
print(type(w_s))
# 1
# <class 'str'>

w_n = int(w_s)
print(w_n)
print(type(w_n))
# 1
# <class 'int'>

w_list = ['日曜日', '月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日']
print(w_list[w_n])
# 月曜日

def get_day_of_week_jp(dt):
    w_list = ['日曜日', '月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日']
    return(w_list[int(dt.strftime('%w'))])

dt2 = datetime.datetime(2018, 1, 2)
print(dt2)
# 2018-01-02 00:00:00

print(get_day_of_week_jp(dt2))
# 火曜日

s = '2018年1月10日'
print(get_day_of_week_jp(datetime.datetime.strptime(s, '%Y年%m月%d日')))
# 水曜日

def get_day_of_week_jp_s(s):
    return get_day_of_week_jp(datetime.datetime.strptime(s, '%Y年%m月%d日'))

print(get_day_of_week_jp_s(s))
# 水曜日

def get_month_jp(dt):
    m_list = [None, '睦月', '如月', '弥生', '卯月', '皐月', '水無月', '文月', '葉月', '長月', '神無月', '霜月', '師走']
    return(m_list[dt.month])

print(get_month_jp(dt2))
# 睦月




