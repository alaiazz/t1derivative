import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


data_root = '/Users/huayuzhu/Desktop/exam/raw_data/daily'
output_dir = '/Users/huayuzhu/Desktop/exam/task1'

# ====================================================================================================================
# # Data Fetching 
# ====================================================================================================================
def get_data_from_csv(file_path):
    """
    Reads a CSV file into a DataFrame, with the first column as row indices, 
    transposes the result, and converts the row indices to datetime.

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
        data = pd.read_csv(file_path, index_col=0).T
        data.index = pd.to_datetime(data.index)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data

# read in datat 
S_DQ_RET = get_data_from_csv(f'{data_root}/S_DQ_RET.csv')
S_905_DQ_RET =  get_data_from_csv(f'{data_root}/905S_DQ_RET.csv')
S_DQ_MV = get_data_from_csv(f'{data_root}/S_DQ_MV.csv')
S_RESTRICT = get_data_from_csv(f'{data_root}/S_RESTRICT.csv')
F7_26 =  get_data_from_csv(f'{data_root}/F7_26.csv')
F7_27 =  get_data_from_csv(f'{data_root}/F7_27.csv')
S_DQ_OPEN = get_data_from_csv(f'{data_root}/S_DQ_OPEN.csv')

# ====================================================================================================================
# # 单因子测试
# ====================================================================================================================
class TestInfo: 
    def __init__(self, group_number = 10, trading_frequency = 'W', initial_capital = None, trading_cost =0.0):
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

class OneFactorTest: 
    """
    A class to conduct single factor testing for stock trading strategies. It ranks stocks, handles trade intervals,
    executes trades, and evaluates performance metrics against a benchmark.

    Attributes
    ----------
    factor : DataFrame
        Factor data used to rank stocks.
    stock_price_open : DataFrame
        Stock opening prices.
    restricted_stock_df : DataFrame
        Information on stocks that are restricted from trading.
    benchmark : DataFrame
        Benchmark data for comparison.

    Methods
    -------
    rank_stock(test_info):
        Ranks stocks based on the provided factor values.

    trade_interval(df, test_info):
        Resamples the given dataframe to the specified trading frequency and forward fills missing data.

    trade(test_info):
        Executes trades based on stock rankings and calculates performance.

    evaluation_metrics(performance, adj1=48, var=0.05):
        Calculates various performance metrics like attribution analysis and risk analysis metrics.

    eval_combined(test_info):
        Evaluates and combines performance metrics for top and bottom ranked stocks.

    compare_with_benchmark(test_info, name, output=False):
        Compares the strategy's performance with a benchmark, returns a dataframe of excess returns.

    plot_comparison_with_benchmark(test_info, name):
        Generates a plot comparing the daily and cumulative excess returns against a benchmark.
    """

    def __init__(self,test_factor):
        min_date = max(test_factor.index.min(), S_DQ_OPEN.index.min())
        max_date = min(test_factor.index.max(), S_DQ_OPEN.index.max())

        self.factor = test_factor.loc[min_date:max_date]
        self.stock_price_open = S_DQ_OPEN.loc[min_date:max_date]
        self.restricted_stock_df = S_RESTRICT
        self.benchmark = S_905_DQ_RET.loc[min_date:max_date]
    
    def rank_stock(self,test_info): 
        factor_value = self.factor.copy()
        percentile = factor_value.rank(axis=1, pct=True)
        group_number = test_info.group_number - ((1 - percentile) * test_info.group_number) // 1
        return group_number
    
    def trade_interval(self,df,test_info):
        df_factor = df.resample(test_info.trading_frequency).first()   
        df_factor = df_factor.ffill() 
        return df_factor

    def trade(self,test_info): 
        df = self.rank_stock(test_info)
        df_price = self.stock_price_open

        # decide initial capital to trade 
        if test_info.initial_capital is not None: 
            initial_k = test_info.initial_capital
        else:
            p_1_top = df.iloc[0] == 1.0
            p_1_bottom = df.iloc[0] == 10.0
            p_1_top_capital = df_price.iloc[0][p_1_top].sum()
            p_1_bottom_capital = df_price.iloc[0][p_1_bottom].sum()
            initial_k = 1000 * (p_1_bottom_capital + p_1_top_capital)
        side_k = initial_k / 2
        top_pnl_tracker = {'total_pnl': [0], 'cumulative_pnl': [0]}
        top_return_tracker = {'return': [0], 'net_return': [0]}
        bottom_pnl_tracker = {'total_pnl': [0], 'cumulative_pnl': [0]}
        bottom_return_tracker = {'net_return': [0]}
        
        factor_resample = self.trade_interval(df,test_info)
        price_resample = self.trade_interval(df_price, test_info)
        top_r = factor_resample.copy() 
        bottom_r = factor_resample.copy() 
        
        if (factor_resample.index != price_resample.index).any():
            print('please debug')
        for i, (date, signals) in enumerate(factor_resample.iterrows()):
            # here equal weighted portfolio strategy is used, Markovitz, or CAPM optimization may give better result  
            top = signals == 1.0
            bottom = signals == 10.0
            top_k = price_resample.loc[date][top].sum()
            bottom_k = price_resample.loc[date][bottom].sum()
            top_ratio = side_k / top_k if top_k else 0
            bottom_ratio = side_k / bottom_k if bottom_k else 0
            top_r.loc[date] = signals.apply(lambda x: top_ratio if x == 1 else 0)
            bottom_r.loc[date] = signals.apply(lambda x: bottom_ratio if x == 1 else 0)

            # profit calculation 
            if i > 0: 
                price_change = price_resample.iloc[i] - price_resample.iloc[i-1]
                
                top_pnl = (top_r.iloc[i-1] * price_change).sum()
                bottom_pnl = (bottom_r.iloc[i-1] * price_change).sum()
        
                top_cumulative_pnl = top_pnl_tracker['cumulative_pnl'][i-1] + top_pnl
                bottom_cumulative_pnl = bottom_pnl_tracker['cumulative_pnl'][i-1] + bottom_pnl

                top_pnl_tracker['total_pnl'].append(top_pnl)
                top_pnl_tracker['cumulative_pnl'].append(top_cumulative_pnl)

                bottom_pnl_tracker['total_pnl'].append(bottom_pnl)
                bottom_pnl_tracker['cumulative_pnl'].append(bottom_cumulative_pnl)

                top_return_tracker['net_return'].append(top_pnl/side_k - test_info.trading_cost)
                bottom_return_tracker['net_return'].append(bottom_pnl/side_k - test_info.trading_cost)
        df_top_pnl_tracker = pd.DataFrame(top_pnl_tracker, index=factor_resample.index)
        df_top_return_tracker = pd.DataFrame(top_return_tracker, index=factor_resample.index)
        df_top_performance = pd.concat([df_top_pnl_tracker, df_top_return_tracker], axis=1)
        df_top_performance['cumulative_return'] = ((df_top_performance['net_return'] + 1).cumprod() -1)/100

        df_bottom_pnl_tracker = pd.DataFrame(bottom_pnl_tracker, index=factor_resample.index)
        df_bottom_return_tracker = pd.DataFrame(bottom_return_tracker, index=factor_resample.index)
        df_bottom_performance = pd.concat([df_bottom_pnl_tracker, df_bottom_return_tracker], axis=1)
        df_bottom_performance['cumulative_return'] = ((df_bottom_performance['net_return'] + 1).cumprod() -1)/100

        return df_top_performance, df_bottom_performance
    
    def evaluation_mectrics(self,performance,adj1 = 48,var=0.05): 
        summary = dict()
        data = performance.loc[:,['net_return']].dropna()
        summary["Annualized Return"] = data.mean() * adj1
        summary["Annualized Volatility"] = data.std() * np.sqrt(adj1)
        summary["Annualized Sharpe Ratio"] = (
            summary["Annualized Return"] / summary["Annualized Volatility"]
        )
        summary["Annualized Sortino Ratio"] = summary["Annualized Return"] / (
            data[data < 0].std() * np.sqrt(adj1)
        )

        summary["Skewness"] = data.skew()
        summary["Excess Kurtosis"] = data.kurtosis()
        summary[f"VaR ({var})"] = data.quantile(var, axis=0)
        summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()
        summary["Min"] = data.min()
        summary["Max"] = data.max()

        wealth_index = 1000 * (1 + data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        summary["Max Drawdown"] = drawdowns.min()

        summary["Bottom"] = drawdowns.idxmin()
        summary["Peak"] = previous_peaks.idxmax()
        return pd.DataFrame(summary) 
    
    def eval_combined(self,test_info): 
        per_top, per_bottom = self.trade(test_info)
        top = self.evaluation_mectrics('top',per_top)
        bottom = self.evaluation_mectrics('bottom',per_bottom)
        return pd.concat([top,bottom], axis=0,keys=['top', 'bottom'])
    
    def compare_with_benchmark(self,test_info,name,output = False):
        per_top, per_bottom = self.trade(test_info)
        ret_top = per_top.loc[:,['net_return']].dropna()
        ret_bottom = per_bottom.loc[:,['net_return']].dropna()
 

        daily_resampled_top = ret_top.resample('D').ffill().div(5)
        daily_resampled_bottom = ret_bottom.resample('D').ffill().div(5)
        daily_resampled_top = daily_resampled_top.reindex(self.benchmark.index, method='ffill')  
        daily_resampled_bottom = daily_resampled_bottom.reindex(self.benchmark.index, method='ffill') 

        daily_resampled_bottom.fillna(0, inplace= True)
        daily_resampled_top.fillna(0, inplace = True)

        if (self.benchmark.index != daily_resampled_bottom.index).any(): 
            print('please debug')
        summary = pd.DataFrame(index = self.benchmark.index) 
        summary["Daily excess_return TOP"] = (daily_resampled_top['net_return'] - self.benchmark[905])
        summary["Cumulative excess_return TOP"] = (1 + summary['Daily excess_return TOP']).cumprod() - 1
        summary["Daily excess_return BOTTOM"] = (daily_resampled_bottom['net_return'] - self.benchmark[905])
        summary["Cumulative excess_return BOTTOM"] = (1 + summary['Daily excess_return BOTTOM']).cumprod() - 1

        if output:
            summary.to_csv(f'{output_dir}/{name}_compairson_ret.csv')

        return summary
    
    def plot_comparison_with_benchmark(self, test_info,name):
        df = self.compare_with_benchmark(test_info,name)
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df['Daily excess_return TOP'], name='Top Daily'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['Daily excess_return BOTTOM'], name='Bottom Daily'),
            secondary_y=False,
        )


        fig.add_trace(
            go.Scatter(x=df.index, y=df['Cumulative excess_return TOP'], name='Top Cumulative'),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['Cumulative excess_return BOTTOM'], name='Bottom Cumulative'),
            secondary_y=True,
        )
       
        fig.update_layout(
            title_text=name+"Comparison of Daily and Cumulative Excess Returns",
            plot_bgcolor='white',  
            xaxis_showgrid=False,  
            yaxis_showgrid=False,  
            yaxis2_showgrid=False,  
            autosize=True,  
            template='plotly_white',
            colorway=px.colors.qualitative.Vivid
        )

        fig.show()


# ====================================================================================================================
# # 因子覆盖率
# ====================================================================================================================

def calc_factor_coverage_rate(df_restrict, df_factor, factor_name):
    """
    Calculates the coverage rate of a factor, indicating the proportion of tradable stocks (not restricted)
    that have non-null factor values.

    Parameters
    ----------
    df_restrict : DataFrame
        DataFrame indicating whether stocks are restricted from trading (1 if restricted, 0 if not).
    df_factor : DataFrame
        DataFrame containing factor values for stocks.
    factor_name : str
        The name to be given to the resulting coverage rate series.

    Returns
    -------
    DataFrame
        A DataFrame with a single column named after `factor_name` containing the coverage rate.
    """

    # check df info 
    if (df_restrict.index != df_factor.index).any(): 
        print('Please debug: length does not match.')
    elif (df_restrict.columns != df_factor.columns).any(): 
        print('Please debug: columns do not match.')
    df_factor_shifted = df_factor.shift(1).iloc[1:]
    df_restrict_aligned = df_restrict.iloc[:-1]
    df_restrict_aligned.index = df_factor_shifted.index
    stock_able_to_trade = (df_restrict_aligned == 0).sum(axis=1)
    factor_cover = ((df_restrict_aligned == 0) & (~df_factor_shifted.isna())).sum(axis=1)
    coverage_rate = factor_cover/stock_able_to_trade
    coverage_rate = coverage_rate.fillna(0)
    return coverage_rate.to_frame(name=factor_name) 

def factor_coverage_rate_plot(df_restrict = S_RESTRICT, df_factor1 = F7_26, df_factor2 = F7_27, 
                              factor_name1= 'F7_26', factor_name2 = 'F7_27', output = True):
    """
    Plots the coverage rates of two factors to compare over time.

    Parameters
    ----------
    df_restrict : DataFrame
        DataFrame indicating whether stocks are restricted from trading.
    df_factor1 : DataFrame
        DataFrame containing the first factor's values.
    df_factor2 : DataFrame
        DataFrame containing the second factor's values.
    factor_name1 : str
        Name of the first factor.
    factor_name2 : str
        Name of the second factor.
    output : bool, default True
        If True, saves the plot as a PNG file. If False, shows the plot interactively.

    Returns
    -------
    None
    """
    coverage_26 = calc_factor_coverage_rate(df_restrict,df_factor1,factor_name1)
    coverage_27 = calc_factor_coverage_rate(df_restrict,df_factor2,factor_name2)

    plt.rcParams['font.size'] = 16
    width = 12
    plt.figure(figsize=(width * 1.41, width))
    plt.plot(coverage_26.index, coverage_26['F7_26'], label='F7_26 Factor Coverage Rate')
    plt.plot(coverage_27.index, coverage_27['F7_27'], label='F7_27 Factor Coverage Rate')

    plt.title('Comparison of Coverage Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Factor Coverage Rate')
    plt.legend()
    if output:
        plt.savefig(f"{output_dir}/coverage_rate_plot.png")
    else:
        plt.show()


# ====================================================================================================================
# # 因子市值中性化 
# ====================================================================================================================


def mad(factor):
    """
    Applies the Median Absolute Deviation (MAD) method to limit the factor values within a threshold to reduce the effect of outliers.

    Parameters
    ----------
    factor : Series or ndarray
        Series or array of factor values.

    Returns
    -------
    Series or ndarray
        Factor values after the MAD method has been applied.
    """
    me = np.median(factor)
    mad = np.median(abs(factor-me))
    up = me + (3*1.4826*mad)
    down = me - (3*1.4826*mad)
    factor = np.where(factor>up,up,factor)
    factor = np.where(factor<down,down,factor)
    return factor

def stand(factor):
    """
    Standardizes the factor values by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    factor : Series or ndarray
        Series or array of factor values.

    Returns
    -------
    Series or ndarray
        Standardized factor values.
    """
    mean = factor.mean()
    std = factor.std()
    return (factor-mean)/std

def neutralize_factors(df_factor, df_MV, name, output = True):
    """
    Neutralizes each factor in df_factor against the market values in df_MV by performing a linear regression and subtracting the predicted values.

    Parameters
    ----------
    df_factor : DataFrame
        DataFrame with factors to be neutralized.
    df_MV : DataFrame
        DataFrame with market values used as independent variables in the regression.
    name : str
        The base name for the output file.
    output : bool, default True
        If True, saves the neutralized factors to a CSV file.

    Returns
    -------
    DataFrame
        DataFrame with neutralized factor values.
    """

    df_f1 = df_factor.copy()
    for col in df_factor.columns:
        factor_col = df_f1[col]
        if factor_col.isnull().all():
            continue
        df_f1[col] = mad(factor_col)
        df_f1[col] = stand(df_f1[col])
        df_f1[col].replace([np.inf, -np.inf], np.nan, inplace=True) 
        
        x = df_MV[col].dropna()
        y = df_f1[col].loc[x.index]   
        valid_idx = y.notna()
        x = x[valid_idx].to_frame()
        y = y[valid_idx]

 
        if not x.empty and not y.isin([np.inf, -np.inf, np.nan]).any():
            L1 = LinearRegression()
            L1.fit(x, y)
            predictions = pd.Series(L1.predict(df_MV[col].dropna().to_frame()), index=df_MV[col].dropna().index)
            df_f1[col].update(df_f1[col] - predictions.reindex(df_f1.index))

    if output:
        df_f1.to_csv(f'{output_dir}/{name}_neutralize.csv')

    return df_f1

if __name__ == '__main__':
    # output the plot of two factor individually and the result of comparison 
    test1 = TestInfo()
    factor27 = OneFactorTest(F7_27)
    #factor27.plot_comparison_with_benchmark(test1,'F7_27')
    factor27.compare_with_benchmark(test1,'F7_27',True)

    factor26 = OneFactorTest(F7_26)
    #factor26.plot_comparison_with_benchmark(test1,'F7_26')
    factor26.compare_with_benchmark(test1,'F7_26',True)

    factor_coverage_rate_plot()
    neutralize_factors(F7_26,S_DQ_MV,'F7_26',output = True)
    neutralize_factors(F7_27,S_DQ_MV,'F7_27',output = True)