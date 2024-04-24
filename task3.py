import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import task1 as t1 
import task2 as t2 

output_dir = '/Users/huayuzhu/Desktop/exam/task3'


# ====================================================================================================================
# # 实现主题因子
# ====================================================================================================================
def illiquidty_factor(open,close,amount,t):
    returns = close/ open.shift(1) - 1
    ret_num = returns.fillna(0)
    amount_den = amount.fillna(0)

    def calc_illiq(x):
        cumulative_product = (1 + abs(x)).prod()
        return np.log(cumulative_product)

    illi_factor_numerator = ret_num.rolling(window=t).apply(calc_illiq, raw=True)
    illi_factor_denominator = amount_den.rolling(window=t).sum()
    illi_factor = illi_factor_numerator / illi_factor_denominator

    return illi_factor[t-1:]

def calculate_time_decayed_illiquidity(close, open, amount,t, decay_factor=0.9):
    """
    Calculates a time-decayed illiquidity factor for each security where recent returns are given more weight.

    Parameters:
    df_returns (DataFrame): DataFrame containing returns with dates as index and securities as columns.
    df_amounts (DataFrame): DataFrame containing trading amounts with dates as index and securities as columns.
    decay_factor (float): The decay factor to apply to historical returns, where 1 is no decay and 0 is full decay.

    Returns:
    DataFrame: Time-decayed illiquidity factors for each security.
    """
    returns = close/ open.shift(1) - 1
    ret_num = returns.fillna(0)
    amount_den = amount.fillna(0)

    def calc_illiq(x):
        weights = np.power(decay_factor, np.arange(len(x))[::-1])
        # Apply weights to the absolute returns
        weighted_abs_returns = abs(x) * weights
        # Calculate the weighted cumulative product
        weighted_cumulative_product = (1 + weighted_abs_returns).prod()
        # Take the log of the cumulative product
        return np.log(weighted_cumulative_product)

    illi_factor_numerator = ret_num.rolling(window=t).apply(calc_illiq, raw=True)
    illi_factor_denominator = amount_den.rolling(window=t).sum()
    illi_factor = illi_factor_numerator / illi_factor_denominator

    return illi_factor[t-1:]

if __name__ == '__main__': 
    illi = illiquidty_factor(t2.open,t2.close,t2.amount,5)
    illi.to_csv(f'{output_dir}/illi_factor.csv')

    open_price = t2.open.resample('D').mean() 
    illi_factor = illi.resample('D').mean() 

    test1 = t1.TestInfo()
    factor3 = t1.OneFactorTest(illi_factor,open_price)
    #factor3.plot_comparison_with_benchmark(test1,'factor3')
    factor3.compare_with_benchmark(test1,'factor3',True)

    time_decayed_illiquidity = calculate_time_decayed_illiquidity(t2.close, t2.open, t2.amount,t=5)
    time_decayed_illiquidity.to_csv(f'{output_dir}/decay_illi_factor.csv')

    deccay_illi_factor = time_decayed_illiquidity.resample('D').mean() 
    factor4 = t1.OneFactorTest(deccay_illi_factor,open_price)
    factor4.compare_with_benchmark(test1,'factor4',True) 