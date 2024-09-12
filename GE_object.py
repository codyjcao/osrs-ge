import osrs_GE

import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns


class GE_object:

    def __init__(self, id,interval = '6h'):
        self.id = id
        self.interval = interval
        self.name = item_name_from_id(self.id)
        self.px_df = osrs_GE.read_item_master_file(self.id,self.interval,create=True)
        
    def check(self):
        return osrs_GE.check(self.px_df)

    def compute_VWAP(self):
        self.px_df = osrs_GE.compute_VWAP(self.px_df)

    def compute_RSI(self,window=14,col_name = 'VWAP'):
        self.px_df = osrs_GE.compute_RSI(self.px_df,window=window,col_name=col_name)

    def compute_SMA(self,window=5,col_name='VWAP'):
        self.px_df = osrs_GE.compute_SMA(self.px_df,window=window,col_name=col_name)

    def compute_MACD(st_n = 4, lt_n = 10, drop_ema_cols = False, col_name = 'VWAP', **kwargs):
        self.px_df = osrs_GE.compute_MACD(self.px_df,st_n,lt_n,drop_ema_cols,col_name,**kwargs)

    def get_item_name(self):
        return self.name
    
    @staticmethod
    def update_item_files(directory='Master Files/items'):
        osrs_GE.update_all_items(directory)
    
    @staticmethod
    def search_item_id():
        osrs_GE.search_item_id()


if __name__ == '__main__':
    GE_object.update_item_files()