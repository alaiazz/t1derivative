import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


data_root = '/Users/huayuzhu/Desktop/exam/raw_data/daily'
output_dir = 'D:/output'
# =============================================================================
# # Data Fetching 
# =============================================================================
def get_data_from_csv(file_path):
    try:
        data = pd.read_csv(file_path, index_col=0).T
        data.index = pd.to_datetime(data.index)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data

S_DQ_RET = get_data_from_csv(f'{data_root}/S_DQ_RET.csv')
S_905_DQ_RET =  get_data_from_csv(f'{data_root}/905S_DQ_RET.csv')
S_DQ_MV = get_data_from_csv(f'{data_root}/S_DQ_MV.csv')
S_RESTRICT = get_data_from_csv(f'{data_root}/S_RESTRICT.csv')
F7_26 =  get_data_from_csv(f'{data_root}/F7_26.csv')
F7_27 =  get_data_from_csv(f'{data_root}/F7_27.csv')
S_DQ_OPEN = get_data_from_csv(f'{data_root}/S_DQ_OPEN.csv')

class test_info: 
    def __init__(self, group_number, trading_frequency , initial_capital, trading_cost):
        """
        Parameters:
        - group_number(int): Number of groups to divide into (default 10 for deciles).
        - trading_frequency(str): "M" (monthly), "W" (weekly) (default to weekly)
        - initial_capital(float): initial capital plan to invest, default to the capital needed at first period trading 
        - trading cost(float): trading cost percentage, default to 0 

        """
        self.group_number = group_number 
        self.trading_frequency = trading_frequency
        self.initial_capital = initial_capital 
        self.trading_cost = trading_cost 

class one_factor_test: 
    def __init__(self,test_factor):
        if test_factor == 'F7_26':
            self.factor = F7_26
        elif test_factor == 'F7_27':
            self.factor == F7_27
        self.stock_price_open = S_DQ_OPEN
        self.restricted_stock_df = S_RESTRICT
        self.benchmark = S_905_DQ_RET
    
    def rank_stock(self,test_info): 
        factor_value = self.factor.copy()
        percentile = factor_value.rank(axis=1, pct=True)
        group_number = test_info.group_number - ((1 - percentile) * test_info.group_number) // 1
        return group_number
    
    def trade(self,test_info): 
        df = self.rank_stock(test_info)
        df_price = self.stock_price_open

        # initial_capital to trade
        if test_info.initial_capital != 0: 
            initial_k = test_info.initial_capital
        else:
            p_1_top = df.iloc[0] == 1
            p_1_bottom = df.iloc[0] == 10
            p_1_top_capital = df_price.iloc[0][p_1_top].sum()
            p_1_bottom_capital = df_price.iloc[0][p_1_bottom].sum()
            initial_k = 10 * (p_1_bottom_capital + p_1_top_capital)
            side_k = initial_k / 2
        
