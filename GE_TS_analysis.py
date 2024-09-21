# -*- coding: utf-8 -*-
"""
Preliminary ARIMA analysis on tradable items in the Grand Exchange
"""

import pandas as pd
import numpy as np


from IPython.display import display

import matplotlib.pyplot as plt

import osrs_GE

from sklearn.metrics import confusion_matrix

# for muting warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


##################################################################

item_id = 6332
freq = '24h'
train_test_split_ratio = .7 # create train test split using 70% for training

print('Conducting analysis for {0} at interval length {1}'.format(
    osrs_GE.item_name_from_id(item_id),freq))

df = osrs_GE.read_item_master_file(item_id,freq)
df.index = pd.DatetimeIndex(df.index.values,freq=df.index.inferred_freq)
df = osrs_GE.compute_VWAP(df)

df['simpRet'] = (df['VWAP']/df['VWAP'].shift(1))-1


df = df.dropna()
train_idx = int(train_test_split_ratio*df.index.shape[0])

# fixing the training set for now, as more data gets added
#train_dt = df.index[train_idx] 
train_dt = '2024-05-19'

df_tr = df[df.index < train_dt]
df_te = df[df.index >= train_dt]

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

# plotting the autocorrelation and partial autocorrelation plots
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 7))
plot_acf(df_tr['simpRet'],lags=10, ax=ax[0])
plot_pacf(df_tr['simpRet'],lags=10, ax=ax[1], method='ols')

plt.tight_layout()
plt.show()

# histogram of daily returns
df_tr['simpRet'].hist()
plt.title('{} daily return'.format(osrs_GE.item_name_from_id(item_id)))
plt.show()


# daily returns over time
df_tr['simpRet'].plot()
plt.title('{} daily return'.format(osrs_GE.item_name_from_id(item_id)))
plt.grid()
plt.show()

# performing ADF test for stationarity
adf_test = adfuller(df_tr['simpRet'])
print('t-stat:{:.3f}'.format(adf_test[0]))
print('p-value:{:.4f}'.format(adf_test[1]))
print('lags:{}'.format(adf_test[2]))

# find optimal ARIMA parameters 
from pmdarima.arima import auto_arima
auto_res = auto_arima(df_tr['simpRet'],seasonal=False,
                      trace=True,
                      max_p=15,
                      max_q=15)

# optimal ARIMA order
P,D,Q = auto_res.order


# iterate through neighbors of the optimal ARIMA order
# evaluating each ARIMA order using the elliptic paraboloid function
res = {}
for p in range(max(1,P-1),P+2):
    for q in range(max(0,Q-1),Q+2):
        order = (p,D,q)
        print(order)
        res[order] = osrs_GE.ARIMA_CV_SCORE(df_tr['simpRet'],
                                            order = order, 
                                            start_point = max(order)+10,
                                            custom_scorer=osrs_GE.elliptic_paraboloid_loss)

print('\n\n')

for key in res:
    print('order = {}'.format(key),
          'EllipPara loss = {}'.format(res[key]['custom_scores'].mean().round(7)),
          'MAE = {}'.format(res[key]['abs_errors'].mean().round(7)),
          'MSE = {}'.format(((res[key]['errors'])**2).mean().round(7)))
    


# finding the 1st order
cur_min = 1e6
cur_min_order=None
for key in res.keys():
    if res[key]['custom_scores'].mean() < cur_min:
        cur_min = res[key]['custom_scores'].mean()
        cur_min_order = key
        
order1 = cur_min_order

# finding the 2nd order
cur_min = 1e6
cur_min_order=None
for key in res.keys():
    if key == order1:
        continue
    if res[key]['custom_scores'].mean() < cur_min:
        cur_min = res[key]['custom_scores'].mean()
        cur_min_order = key
        
order2 = cur_min_order

orders = [order1,order2]


# define the thresholds for the trading signal?
# find a way to automatically set the thresholds based on the volatility of the product
# or based on the training data calibrated to the test set?
Q_s, Q_b = (-.0075,.005)

# inventory limit, might be some cleverer way of auto-setting this depending
# on the actual price of the item
inv_limit = 500
    

for order in orders:
    print(order)
    df_res = res[order]['ret_df']
    
    df_res['signal'] = 1*(df_res['Pred'] >= Q_b) - 1*(df_res['Pred'] <= Q_s)
    df_res['actual'] = 1*(df_res['Real'] >= Q_b) - 1*(df_res['Real'] <= Q_s)

    df_res = df_res.merge(df_tr, how='outer',left_index=True,right_index=True)
    df_res['VWAP_trade'] = df_res['VWAP'].shift(1)
    df_res = df_res.dropna()
    df_res = df_res.drop(['timestamp','highPriceVolume','lowPriceVolume'],axis=1)
    df_res.index = df_res.index.shift(-1,df_res.index.inferred_freq)

    display(pd.DataFrame(confusion_matrix(df_res['actual'],df_res['signal']),
                        index = ['sell','hold','buy'],columns=['sell','hold','buy']))
    
    pnl_result = osrs_GE.trading_strategy_pnler(df_res,inv_limit,start_stack = start_stack)

    print("Total PnL: {}".format((pnl_result.tail(1)['total_portfolio'].values[0] - start_stack).round()))
    print("Final inventory: {}".format(int(pnl_result.tail(1)['inventory'].values[0])))
    print('\n')