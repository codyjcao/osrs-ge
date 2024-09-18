import requests
import json

import os
import re

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from datetime import datetime
from osrsbox import items_api

####################################################################################

items = items_api.load()

col_names = ['name','lowalch','highalch']
MAST_ITEM_DF = pd.DataFrame(columns=col_names)

for item in items:
    if item.tradeable_on_ge and not item.noted:
        MAST_ITEM_DF.loc[item.id] = (item.name,item.lowalch,item.highalch)

####################################################################################

common_trade_idx_url = "https://oldschool.runescape.wiki/w/RuneScape:Grand_Exchange_Market_Watch/Common_Trade_Index"

html = requests.get(common_trade_idx_url).content

df_index = pd.DataFrame((pd.read_html(html)[0])['Item.1'])
df_index = df_index.rename({'Item.1':'name'},axis=1)

df_index['id'] = df_index['name'].apply(lambda x:MAST_ITEM_DF.index[MAST_ITEM_DF['name'] == x][0])


####################################################################################

item_name_from_id = lambda x: items.lookup_by_item_id(x).name

####################################################################################



####################################################################################

def call_http_historical_prices(item_id = 1,interval = '24h'):
    headers = {
        'User-Agent': 'Flipping Price',
        'From': 'email@gmail.com'
    }
    response = requests.get('https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={1}&id={0}'.format(item_id,interval),headers=headers)
    stats = json.loads(response.text)
    return stats


def create_price_df(item_id,interval='24h'):
    jsonData = call_http_historical_prices(item_id,interval)
    data = pd.DataFrame(jsonData)
    if data.shape[0] == 0:
        raise HistoricalDataNotFound('This item id has no historical data available')

    if data.shape[0] < 365:
        print("DataFrame has fewer than 365 rows; some data may be missing: ", item_id)

    data['index_id'] = np.arange(len(data))
    temp = data['data']
    res = pd.json_normalize(temp)
    
    res['id'] = item_id
    res['date'] = pd.to_datetime(res['timestamp'],unit='s')
    return res


def read_CTI_master_file(interval='24h',create=True):
    filepath = 'Master Files/CTI_master_file_{}.csv'.format(interval)
    save = False
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        if create:
            df = pd.DataFrame()
            print("File does not exist, creating from scratch for interval {0}".format(interval))
            for row in df_index.itertuples(index=False):
                try:
                    price_history = create_price_df(row[1],interval)
                except:
                    print('Skipping item id {0}'.format(item_id))
                    continue
                price_history.fillna(0)
                df = pd.concat([df,price_history])
            save = True
        else:
            print("Returning nothing, specify create == True to create dataframe from scratch")
            return None
    except:
        print("Something went wrong")
            
    df['date'] = pd.to_datetime(df['date'])    
    df.index = pd.MultiIndex.from_frame(df[['id','date']])
    df.index = df.index.set_levels(pd.DatetimeIndex(df.index.levels[1].values,
                                                    freq=df.index.levels[1].inferred_freq),level=1)
    df.drop(['id','date'],axis=1,inplace=True)
    df = df.sort_index()
    if save:
        df.to_csv(filepath)
        print("Saving file to {0}".format(filepath))
    return df

def update_CTI_master_file(interval='24h',return_df = True):
    df = read_CTI_master_file(interval=interval)
    df_add = pd.DataFrame()
    
    #for item_id, new_df in df.groupby(level='id'):
    for item_id in df_index['id']:
        item_name = item_name_from_id(item_id)
        print('Retrieving data for item {0} ({1})'.format(item_id, item_name))

        try:
            temp_df = create_price_df(int(item_id),interval)
        except:
            print('Item {0} ({1}) retrieval failed...'.format(item_id, item_name))
            continue
        temp_df.fillna(0)
        df_add = pd.concat([df_add,temp_df])
        print('Item {0} ({1}) data retrieved successful'.format(item_id,item_name))

    df_add.index = pd.MultiIndex.from_frame(df_add[['id','date']])
    df_add.drop(['id','date'],axis=1,inplace=True)
    
    df = pd.concat([df,df_add]).drop_duplicates(keep='last').sort_index()
    df.to_csv('Master Files/CTI_master_file_{}.csv'.format(interval))
    print("Saving file... Master Files/CTI_master_file_{}.csv".format(interval))
    if return_df:
        return df

def compute_CTI(df=None,interval='24h'):
    if df is None:
        df = read_CTI_master_file(interval=interval)
        df = compute_VWAP(df)
    
    df_index_px = pd.read_excel('index_start_px.xlsx',index_col='id',names = ['name','id','start_px'])
    df_index_px['weight'] = 1/df_index_px['start_px']
    INDEX_MULT = 100.0/df_index_px.shape[0]
    df = df.filter(['VWAP'])
    earliest = df.groupby(by='id').apply(lambda x:x.index.get_level_values(1).min()).max()
    df = df.loc[df.index.get_level_values(1)>=earliest]

    df = pd.merge(df,df_index_px,how='left',left_on='id',right_index=True)
    
    df['index_ctrb'] = df['weight'] * df['VWAP']
    CTI_px = df.groupby('date')['index_ctrb'].sum()*INDEX_MULT
    return CTI_px.rename('CTI_px')

    

def update_item_master_file(item_id, interval='24h',return_df = True):
    file_path = 'Master Files/items/master_file_{0}_{1}.csv'.format(item_id,interval)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('File does not exist at {0}'.format(file_path))
        print('Returning none...')
        return None
    
    df.index = df.date
    df = df.drop('date',axis=1)
    try:    
        df_add = create_price_df(item_id,interval)
    except:
        print('Unable to load new data for parameters {0}, {1}'.format(item_id,interval))
        return df
    df_add.index = df_add.date
    df_add = df_add.drop(['date','id'],axis=1)
    
    df = pd.concat([df,df_add]).drop_duplicates(keep='last')
    df.to_csv(file_path)
    print('Updated file and saved to {0}'.format(file_path))
    if return_df:
        return df


def read_item_master_file(item_id,interval='24h',create = True):
    file_path = 'Master Files/items/master_file_{0}_{1}.csv'.format(item_id,interval)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('File does not exist for {0}'.format(file_path))
        if create:
            try:
                df = create_price_df(item_id,interval)
            except:
                print('Unable to create dataframe for {0}, {1}'.format(item_id,interval))
                return None
            df.index = df.date
            df = df.drop(['id','date'],axis=1)
            print('Saving to {0}'.format(file_path))
            df.to_csv(file_path)
            return df
        print('Specify create = True to create file')
        return None
    print("File read successfully for {0}".format(file_path))
    df['date'] = pd.to_datetime(df['timestamp'],unit='s')
    
    df.index = df.date

    df.index.freq = df.index.inferred_freq
    
    return df.drop('date',axis=1)


def check(df):
    to_check_filter = (
        (df['lowPriceVolume'] == 0) |
        (df['highPriceVolume'] == 0) | 
        (df['avgLowPrice'].isnull()) |
        (df['avgHighPrice'].isnull()))
    return df[to_check_filter],df[to_check_filter].index


def update_all_items(directory = 'Master Files/items'):
    dir = directory
    for filename in os.listdir(dir):
        f = os.path.join(dir,filename)
        if os.path.isfile(f):
            id,interval = re.split("\W+|_",f)[-3:-1]
            id = int(id)
            update_item_master_file(id,interval,return_df=False)

def search_item_id():
    search_term = input("Enter search term: ")
    results = []
    for item_name in MAST_ITEM_DF['name'].values:
        if search_term.lower() in item_name.lower():
            results.append((item_name,MAST_ITEM_DF.index[MAST_ITEM_DF['name'] == item_name]))
    
    if results:
        print("Search results:")
        for result in results:
            print(result)
    else:
        print("No results found.")

#####################################################################
# Price features for 
## VWAP
def compute_VWAP(df,high_p_name = 'avgHighPrice',low_p_name = 'avgLowPrice',low_v_name = 'lowPriceVolume', high_v_name = 'highPriceVolume'):
    df['VWAP'] = ((df['avgHighPrice'] * df['highPriceVolume'] + df['avgLowPrice'] * df['lowPriceVolume'])/
                 (df['highPriceVolume']+df['lowPriceVolume']))
    return df

## Moving averages
### Simple moving average
def compute_SMA(df,window = 5,col_name = 'VWAP'):
    df[col_name + '_sma'+str(window)] = df[col_name].rolling(window=window).mean()
    return df

## RSI
def _helper_RSI(x):
    if x[x>0].shape[0]:
        avgU = x[x>0].mean()
    else:
        avgU = 0

    if x[x<0].shape[0]:
        avgD = abs(x[x<0].mean())
    else:
        return 100
    RS = avgU/avgD
    return 100 - ( 100 / (1 + RS) )


def compute_RSI(df,window=14,col_name = 'VWAP'):
    df[col_name + '_diff'] = df[col_name].diff()
    df[col_name + '_RSI'+str(window)] = df[col_name + '_diff'].rolling(window=window).apply(_helper_RSI)
    return df


## MACD
def compute_MACD(df,st_n = 4,lt_n = 10,drop_ema_cols = False,col_name = 'VWAP',**kwargs):
    if st_n >= lt_n:
        print('Invalid choice of short and long term periods. st_n < lt_n is required')
        return df

    st_ema_col_name = col_name + '_ema'+str(st_n)
    lt_ema_col_name = col_name + '_ema'+str(lt_n)
    
    df[st_ema_col_name] = df[col_name].ewm(span=st_n, **kwargs).mean()
    df[lt_ema_col_name] = df[col_name].ewm(span=lt_n, **kwargs).mean()
    
    df[col_name + '_MACD_{0}_{1}'.format(str(st_n),str(lt_n))] = df[st_ema_col_name] - df[lt_ema_col_name]

    if drop_ema_cols:
        return df.drop([st_ema_col_name, lt_ema_col_name],axis=1)
    
    return df


###########################################################################

def return_last_n_days(df,n_days,col_name = 'VWAP',log_ret=False):
    # price data frame with datetime index and VWAP column
    df = df.sort_index()
    n_days_ago = df.index.levels[1].sort_values().unique()[-1] - pd.Timedelta(n_days,'D')
    fltr = df.index.get_level_values(1)>=n_days_ago
    last_n_df = df.loc[fltr]
    if log_ret:
        last_n_ret = last_n_df.groupby('id')[col_name].agg(lret = lambda x:np.log(x.iloc[-1]/x.iloc[0])).sort_values(by='lret')
        last_n_ret = last_n_ret.rename(axis=1,mapper = {'lret':str(n_days)+'_days_lret'})
        last_n_ret['name'] = pd.Series(last_n_ret.index.values).apply(item_name_from_id).values
        return last_n_ret
    last_n_ret = last_n_df.groupby('id')[col_name].agg(ret=lambda x:(x.iloc[-1]/x.iloc[0]) - 1).sort_values(by='ret')
    last_n_ret = last_n_ret.rename(axis=1,mapper={'ret':str(n_days)+'_days_ret'})
    last_n_ret['name'] = pd.Series(last_n_ret.index.values).apply(item_name_from_id).values
    return last_n_ret

def return_between(df,date1,date2,col_name = 'VWAP',log_ret = False):
    if date1 == date2:
        raise InvalidDateException("Dates are the same")
    later = max(date1,date2)
    early = min(date1,date2)
    
    fltr = (df.index.get_level_values(1) <= later) & (df.index.get_level_values(1) >= early)
    fltr_df = df.loc[fltr]
    
    if log_ret:
        df_ret = fltr_df.groupby('id')[col_name].agg(lret = lambda x:np.log(x.iloc[-1]/x.iloc[0])).sort_values(by='lret')
        df_ret['name'] = pd.Series(df_ret.index.values).apply(item_name_from_id).values
        return df_ret
    df_ret = fltr_df.groupby('id')[col_name].agg(ret=lambda x:(x.iloc[-1]/x.iloc[0]) - 1).sort_values(by='ret')
    df_ret['name'] = pd.Series(df_ret.index.values).apply(item_name_from_id).values
    return df_ret
   
    
    
from sklearn.linear_model import LinearRegression

def _compute_beta(series,index,print_errors = False):
    try:
        s1 = pd.Series(series.values,index=series.index.get_level_values(1),name='s')
    except:
        s1 = pd.Series(series,name='s')
    s2 = pd.Series(index,name='idx')
    df = pd.concat([s1,s2],axis=1)
    if not df[df.isna().any(axis=1)].empty and print_errors:
        print(series.index.get_level_values(0)[0])
        print(df[df.isna().any(axis=1)])
    
    df = df.dropna()
    
    model = LinearRegression(fit_intercept=False).fit(df['idx'].values.reshape(-1,1),df['s'])
    return model.coef_[0]


def compute_betas(df,n,print_missing=False,print_errors=False):
    # compute betas over different time frames
    start_date = datetime.today() - pd.Timedelta(n,'D')
    
    if type(df.index) == pd.core.indexes.multi.MultiIndex:
        CTI_idx = compute_CTI(df = df)
        CTI_idx_ret = (CTI_idx/CTI_idx.shift(1)) - 1
        #start_date = df.groupby('id')['simpRet'].agg(latest = lambda x:x.index.levels[1].max()).min().values[0] - pd.Timedelta(n,'D')
        df_fltr = df[df.index.get_level_values(1) >= start_date]
        CTI_idx_ret = CTI_idx_ret[CTI_idx_ret.index >= start_date]
        nrows = df_fltr.groupby('id')['simpRet'].agg(nrow=lambda x:x.shape[0])
        max_days = nrows.max().values[0]
        if print_missing:
            print(nrows[nrows['nrow']!= max_days])
        res = df_fltr.groupby('id')['simpRet'].agg(beta = lambda x:_compute_beta(x,CTI_idx_ret,print_errors))
        res['name'] = pd.Series(res.index.values).apply(item_name_from_id).values
        res = (res.rename({'beta':'beta_'+str(n)},axis=1)).sort_values(by='beta_'+str(n))
    else:
        CTI_idx = compute_CTI()
        CTI_idx_ret = (CTI_idx/CTI_idx.shift(1)) - 1
        df_fltr = df[df.index >= start_date]
        CTI_idx_ret = CTI_idx_ret[CTI_idx_ret.index >= start_date]
        res = _compute_beta(df_fltr['simpRet'],CTI_idx_ret,print_errors)
        
    return res
        
    
    
def rolling_betas(item_id,window_size = 30,interval='24h',plot=False):
    CTI_idx = compute_CTI(interval=interval)
    CTI_idx_ret = (CTI_idx/CTI_idx.shift(1)) - 1
    
    if item_id == 0:
        # debugging
        CTI_series = compute_CTI(interval=interval)
        item_ret_series = (CTI_series/CTI_series.shift(1))-1
    else:
        df_item = compute_VWAP(read_item_master_file(item_id,interval))
        item_ret_series = (df_item['VWAP']/df_item['VWAP'].shift(1))-1
    
    
    CTI_idx_ret = CTI_idx_ret.dropna()
    item_ret_series = item_ret_series.dropna()
    df = pd.DataFrame({'CTI_ret':CTI_idx_ret,str(item_id)+'_ret':item_ret_series}).dropna()
    rolling = df[['CTI_ret']].rolling(window=window_size,min_periods=window_size)
    res = ((rolling.cov(df[str(item_id)+'_ret']))/rolling.var()).dropna().rename({'CTI_ret':'beta_'+str(window_size)},axis=1)
    if plot:
        res.plot()
        plt.grid()
        plt.title('Rolling {}-day beta'.format(window_size))
        plt.show()
    
    return res

######################################################################################################


from statsmodels.tsa.arima.model import ARIMA

def ARIMA_CV_SCORE(series,order = (1,0,0),start_point = 10,custom_scorer=None):
    # custom_scorer is a function that takes in two variables and returns a single scalar value
    custom_score = custom_scorer is not None
    errors = []
    conv_issues = []
    preds = []
    custom_scores = []
    for k in range(start_point,series.shape[0]-1):
        model = ARIMA(series.iloc[:k],order=order)
        model_fit = model.fit(method_kwargs={'maxiter':100})
        pred = (model_fit.forecast()).values[0]
        obs = series.iloc[k+1]
        errors.append(pred - obs)
        preds.append(pred)
        if custom_score:
            custom_scores.append(custom_scorer(pred,obs))
        #print(k)
    ret_df = pd.DataFrame(index = series.index[start_point:-1],data = {'Pred':preds,'Real':series.iloc[start_point:-1]})
    errors = np.array(errors)
    
    if custom_score:
        return {'errors':errors, 'abs_errors':abs(errors),'ret_df':ret_df,'custom_scores':np.array(custom_scores)}
    
    return {'errors':errors, 'abs_errors':abs(errors),'ret_df':ret_df}


######################################################################################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def elliptic_paraboloid_loss(x, y, c_diff_sign=4, c_same_sign=0.1):
    # Compute a rotated elliptic parabaloid.
    t = np.pi / 4
    x_rot = (x * np.cos(t)) + (y * np.sin(t))
    y_rot = (x * -np.sin(t)) + (y * np.cos(t))
    z = ((x_rot**2) / c_diff_sign) + ((y_rot**2) / c_same_sign) 

    return(z)

def elliptic_paraboloid_loss_obj(B_vec, x_mat, y_obs,c_diff_sign = 8,c_same_sign = 2):
    # x_mat is nxp
    # B_mat is px1
    # y_obs is nx1
    n = x_mat.shape[0]
    p = x_mat.shape[1]

    B_vec = B_vec.reshape(p,1)
    y_obs = y_obs.reshape(n,1)

    return elliptic_paraboloid_loss(np.matmul(x_mat,B_vec),y_obs,c_diff_sign,c_same_sign).sum()

######################################################################################################

def compute_n_simple_return(df,n=1,col_name = 'VWAP'):
    # computes the lookahead return n periods from now
    if n==1:
        ret_col_name = col_name + '_ret_la'
    else:
        ret_col_name = col_name + '_ret_la_'+str(n)
    
    df[ret_col_name] = (df[col_name].shift(-n)/df[col_name]) - 1
    return df    

######################################################################################################

def trading_strategy(df,max_allowable = 2,start_stack = 0, signal_column = 'signal',price_column = 'VWAP_trade'):
    # df should have the signal and VWAP columns already
    inv = 0
    stack = start_stack
    
    trading_history = pd.DataFrame()
    
    for idx,row in df.iterrows():
        if row[signal_column] == 1 and inv < max_allowable:  # buy signal and we are not at inventory capacity
            stack -= row[price_column]
            inv += 1
        elif row[signal_column] == -1 and inv > 0:       # sell signal and we have positive inventory
            stack += row[price_column]
            inv -= 1
        else:
            pass
        
        trading_history.loc[idx,'inventory'] = inv
        trading_history.loc[idx,'stack'] = stack
        trading_history.loc[idx,'total_portfolio'] = stack + inv*row[price_column]
    
    trading_history['total_portfolio'].plot()
    plt.grid()
    plt.title('Portfolio value + GP')
    plt.show()

    return trading_history



if __name__ == '__main__':
    ####
    ####
    print(0)