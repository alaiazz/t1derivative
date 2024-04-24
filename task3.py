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

