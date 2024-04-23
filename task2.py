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
from scipy.stats import spearmanr

# ====================================================================================================================
# # Data Fetching 
# ====================================================================================================================
data_root_min = '/Users/huayuzhu/Desktop/exam/raw_data/minute'
data_root = '/Users/huayuzhu/Desktop/exam/raw_data/daily'
output_dir = '/Users/huayuzhu/Desktop/exam/task2'

def get_min_data_from_csv(file_path):
    """
    Reads a CSV file into a DataFrame, with the first column as row indices.

    Parameters
    ----------
    file_path : str
        The file path of the CSV file to be read.

    Returns
    -------
    pd.DataFrame
        Transposed DataFrame with datetime as index.
    """
    try:
        data = pd.read_csv(file_path, index_col=0) 
        data.index = pd.to_datetime(data.index)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data

amount = get_min_data_from_csv(f'{data_root_min}/amount.csv')
volume = get_min_data_from_csv(f'{data_root_min}/volume.csv')
close  = get_min_data_from_csv(f'{data_root_min}/close.csv')
open  = get_min_data_from_csv(f'{data_root_min}/open.csv')

S_DQ_RET = t1.get_data_from_csv(f'{data_root}/S_DQ_RET.csv')
S_905_DQ_RET =  t1.get_data_from_csv(f'{data_root}/905S_DQ_RET.csv')
S_DQ_MV = t1.get_data_from_csv(f'{data_root}/S_DQ_MV.csv')
S_RESTRICT = t1.get_data_from_csv(f'{data_root}/S_RESTRICT.csv')
S_DQ_OPEN = t1.get_data_from_csv(f'{data_root}/S_DQ_OPEN.csv')
S_DQ_ADJ_FACTOR = t1.get_data_from_csv(f'{data_root}/S_DQ_ADJFACTOR.csv')
S_DQ_CLOSE = t1.get_data_from_csv(f'{data_root}/S_DQ_CLOSE.csv')
S_DQ_VOLUME = t1.get_data_from_csv(f'{data_root}/S_DQ_VOLUME.csv')

S_DQ_ADJ_FACTOR = S_DQ_ADJ_FACTOR/100
S_ADJ_CLOSE = S_DQ_CLOSE * S_DQ_ADJ_FACTOR
S_ADJ_OPEN = S_DQ_OPEN * S_DQ_ADJ_FACTOR

# ====================================================================================================================
# # 实现因子函数
# ====================================================================================================================
def calculate_factor(df_adj_close, df_adj_open, df_volume, df_market_value, N):
    """
    Calculates the mean overnight return for the days where the past N days' turnover rates
    are in the top and bottom 20%.

    Parameters
    ----------
    df_adj_close : DataFrame
        Adjusted closing prices with dates as index and tickers as columns.
    df_adj_open : DataFrame
        Adjusted opening prices with dates as index and tickers as columns.
    df_volume : DataFrame
        Trading volume with dates as index and tickers as columns.
    df_market_value : DataFrame
        Market value with dates as index and tickers as columns.
    N : int
        The number of days to look back for turnover rates.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the calculated mean overnight returns.
    """
    overnight_returns = df_adj_open / df_adj_close.shift(1) - 1
    turnover_rates = df_volume / df_market_value
    
    df = pd.concat([overnight_returns,turnover_rates], keys = ['OverNight','TurnOver'],axis = 1)
    def calc_within_window(window_df):
        turnover_window = window_df.loc[:, 'TurnOver']
        overnight_window = window_df.loc[:, 'OverNight']
        ranked_turnover = turnover_window.rank(pct=True)
        is_top_20 = ranked_turnover >= 0.8
        is_bottom_20 = ranked_turnover <= 0.2
        mean_over_night = overnight_window[is_top_20 | is_bottom_20].mean()
        
        return mean_over_night
    df_factor = pd.DataFrame(index = df.index[N:], columns = overnight_returns.columns)
    # this is not applied for the large dataset, but due to time limit, I used for loop here for simplicity 
    for i in range(len(df_factor)-1): 
        df_factor.iloc[i] = calc_within_window(df[i:N+i])
    return df_factor 


def calculate_adjusted_turnover_weighted_returns(df_adj_close, df_adj_open, df_volume, df_market_value, N):
    """
    Calculates the weighted mean overnight returns for stocks based on turnover rates from the past N days.
    The weighted mean is computed by doubling the mean overnight return for the top 20% of turnover rates
    and halving the mean overnight return for the bottom 20%, then summing these two values.

    Parameters
    ----------
    df_adj_close : pd.DataFrame
        DataFrame containing adjusted closing prices with dates as index and securities (e.g., tickers) as columns.
    df_adj_open : pd.DataFrame
        DataFrame containing adjusted opening prices with dates as index and securities as columns.
    df_volume : pd.DataFrame
        DataFrame of trading volumes for each security with dates as index and securities as columns.
    df_market_value : pd.DataFrame
        DataFrame of market values for each security with dates as index and securities as columns.
    N : int
        Number of days to use in the rolling window to determine the top and bottom 20% of turnover rates.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each entry corresponds to the weighted mean overnight return for each ticker. 
    """
    overnight_returns = df_adj_open / df_adj_close.shift(1) - 1
    turnover_rates = df_volume / df_market_value
    
    df = pd.concat([overnight_returns,turnover_rates], keys = ['OverNight','TurnOver'],axis = 1)
    def calc_within_window(window_df):
        turnover_window = window_df.loc[:, 'TurnOver']
        overnight_window = window_df.loc[:, 'OverNight']
        ranked_turnover = turnover_window.rank(pct=True)
        is_top_20 = ranked_turnover >= 0.8
        is_bottom_20 = ranked_turnover <= 0.2        
        mean_over_night_top = overnight_window[is_top_20].mean() * 2 
        mean_over_night_bottom = overnight_window[is_bottom_20].mean() / 2 
        mean_over_night = (mean_over_night_top + mean_over_night_bottom)
        return mean_over_night 

        
    df_factor = pd.DataFrame(index = df.index[N:], columns = overnight_returns.columns)
    # this is not applied for the large dataset, but due to time limit, I used for loop here for simplicity 
    for i in range(len(df_factor)-1): 
        df_factor.iloc[i] = calc_within_window(df[i:N+i])
    return df_factor 


# ====================================================================================================================
# # rankIC & IR 
# ====================================================================================================================
def calc_rankIC(factor, price_df, N): 
    """
    Calculates the rank IC for each security.

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame containing factor scores with dates as index and securities as columns.
    price_df : pd.DataFrame
        DataFrame containing prices with dates as index and securities as columns.
    N : int
        Number of days used in calculating Rank IC to compute returns.
    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row containing the ICIR for each security.
    """
     
    ret = (price_df.shift(-N)/price_df - 1)[:-5]
    factor = factor[:-5]
    rankIC = pd.DataFrame(index=factor.resample('M').mean().index)

    for ticker in factor.columns:
        combined = pd.concat([factor[ticker], ret[ticker]], axis=1, keys=['factor', 'returns']) 
        combined.dropna(inplace=True)
        if combined.empty:
             rankIC[ticker] = pd.Series(pd.NA, index=rankIC.index) 
        else:
            monthly_corr = combined.resample('M').apply(lambda x: x['factor'].corr(x['returns'], method='spearman'))
            rankIC[ticker] = monthly_corr 
    return rankIC

def calc_ICIR(factor, price_df, N):
    """
    Calculates the Information Coefficient Information Ratio for each security.

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame containing factor scores with dates as index and securities as columns.
    price_df : pd.DataFrame
        DataFrame containing prices with dates as index and securities as columns.
    N : int
        Number of days used in calculating Rank IC to compute returns.
    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row containing the ICIR for each security.
    """
     
    rankIC = calc_rankIC(factor, price_df, N)
    ICIR = rankIC.mean(axis = 0)/ rankIC.std(axis =0)
    ICIR = ICIR.to_frame()
    return ICIR.T 

# ====================================================================================================================
# # vwap calculation 
# ====================================================================================================================
def calc_VWAP(amount, volume, N):
    """
    Calculates VWAP for each security.

    Parameters
    ----------
    amount : pd.DataFrame
    volume : pd.DataFrame
    N : int
    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row containing the VWAP for each security.
    """

    # data slicing 
    amount = amount.between_time('13:01', '14:00') #成交额
    volume = volume.between_time('13:01', '14:00') #成交量
    daily_amount = amount.resample('D').sum()
    daily_volume = volume.resample('D').sum()
    if N == 1:
        vwap = daily_amount/ daily_volume
    else: 
        rolling_amount = daily_amount.rolling(window=N).sum()
        rolling_volume = daily_volume.rolling(window=N).sum()
        vwap = rolling_amount / rolling_volume
    return vwap[N:]


if __name__ == '__main__': 

    # factor 1 construction 
    df_factor = calculate_factor(S_ADJ_CLOSE, S_ADJ_OPEN,S_DQ_VOLUME , S_DQ_MV, N=20)[:-1]
    factor_2016 = df_factor.loc['2016':]
    factor_2016.to_csv(f'{output_dir}/factor1.csv')

    # calculate ICIR 
    ICIR = calc_ICIR(df_factor.loc['2016':],S_DQ_OPEN.loc['2016':],5)
    ICIR.to_csv(f'{output_dir}/ICIR.csv')

    # test factor1 with benchmark 
    test1 = t1.TestInfo()
    factor1 = t1.OneFactorTest(df_factor)
    factor1.compare_with_benchmark(test1,'Factor1',True)

    # factor 1.2 construction 
    df_factor_2 = calculate_adjusted_turnover_weighted_returns(S_ADJ_CLOSE, S_ADJ_OPEN,S_DQ_VOLUME , S_DQ_MV, N=20)[:-1]
    factor_2_2016 = df_factor.loc['2016':]
    factor_2_2016.to_csv(f'{output_dir}/factor1.2.csv')

    # test factor1.2 with benchmark 
    factor1_2 = t1.OneFactorTest(df_factor_2)
    factor1_2.compare_with_benchmark(test1,'Factor1.2',True)

    # N = {1,5,10}, calculate VWAP 
    vwap_1 = calc_VWAP(amount,volume, 1)
    vwap_5 = calc_VWAP(amount,volume, 5)
    vwap_10 = calc_VWAP(amount,volume, 10)
    vwap_1.to_csv(f'{output_dir}/vwap_1.csv')
    vwap_5.to_csv(f'{output_dir}/vwap_5.csv')
    vwap_10.to_csv(f'{output_dir}/vwap_10.csv')






    
    