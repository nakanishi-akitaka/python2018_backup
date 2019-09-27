# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
Created on Thu Oct 25 14:43:33 2018

@author: Akitaka
"""

#%%
from datetime import datetime
datetime(year=2015, month=7, day=4)

from dateutil import parser
date = parser.parse("4th of July, 2015")
date
date.strftime('%A')

## Typed arrays of times: NumPy's datetime64
import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
date
date + np.arange(12)
np.datetime64('2015-07-04')
np.datetime64('2015-07-04 12:59:59.50', 'ns')

## Dates and times in pandas: best of both worlds
import pandas as pd
date = pd.to_datetime("4th of July, 2015")
date
date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')

# Pandas Time Series: Indexing by Time
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data
data['2014-07-04':'2015-07-04']
data['2015']

# Pandas Time Series Data Structures
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
dates
dates.to_period('D')
dates - dates[0]


## Regular sequences: pd.date_range()
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8)
pd.date_range('2015-07-03', periods=8, freq='H')
pd.date_range('2015-07', periods=8, freq='M')

pd.timedelta_range(0, periods=10, freq='H')

# Frequencies and Offsets
pd.timedelta_range(0, periods=9, freq="2H30T")

from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())

# Resampling, Shifting, and Windowing
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016',
                       data_source='google')
goog.head()