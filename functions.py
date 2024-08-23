# Importacao das libraries
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


################################################################################
#                               START HERE                                     #

'''
from MyCustomLibrary.main import * # import do ficheio main da pasta My Custom Library
tickers =["BTC-USD", "ETH-USD", "SOL-USD"]

# Análise personalizada
start_date = "2023-01-01"
end_date = "2024-01-01"

rf = 0.035 #assuming a 3.5% risk free rate
days_per_year = 365 # 365 for crypto | 252 for stocks



quotes = mcl.download_yahoo_data(tickers, start_date, end_date)

quotes_semanal = quotes.resample('W').last()
quotes_mensal = quotes.resample('BM').last()
quotes_trimestral = quotes.resample('Q').last()
quotes_anual = quotes.resample('A').last()
'''


################################################################################
#                          FUNÇOES DIVERSAS                                    #

def reload(mcl):
    importlib.reload(mcl)


def merge_time_series(df_1, df_2, how='outer'):
    df = df_1.merge(df_2, how=how, left_index=True, right_index=True)
    return df


def normalize(df):
    df = df.dropna()
    return (df / df.iloc[0]) * 100


def save_csv(df):
    df.to_csv(f"{df}.csv", date_format='%Y-%m-%d')


def read_csv(filepath):
   quotes = pd.read_csv(f"r{filepath}", delimiter=',')
   quotes.set_index('Date', inplace=True)
   return quotes

def compute_years(df, days_per_year="days_per_year"):
    years = (df.index[-1] - df.index[0]).days / days_per_year
    return round(years,1)


################################################################################
#                              DICTIONAIRES                                    #

frequency_map = {
        "daily": 365.25,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1}

################################################################################
#                               GET DATA                                       #


#######--------------YAHOO FINANCE---------------#######
def download_yahoo_data(tickers, start, end, interval="1d"):
    quotes = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start, end, interval, progress=False)# , interval = 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        df = df[['Adj Close']]
        df.columns=[ticker]
        quotes = merge_time_series(quotes, df)

    quotes = quotes.ffill() # fill missing values in a DataFrame or Series by forward filling them.

    return quotes

#######--------------BINANCE---------------#######

from binance.client import Client
client = Client()

def download_binance_ohlc(ticker, start, end, interval="1d"): 
    # interval = "1s", "1m", "5m", "15m", "30m", 1h", "4h", "8h", "12h", "1d", "1w", "1M" 
    df = pd.DataFrame(client.get_historical_klines(ticker, interval, start, end))
    df = df.iloc[:,:6]
    df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Time", inplace=True)
    df.index = pd.to_datetime(df.index, unit="ms")
    df = df.astype(float)
    return df

def download_binance_data(tickers, start, end, interval):
    # interval = "1s", "1m", "5m", "15m", "30m", 1h", "4h", "8h", "12h", "1d", "1w", "1M" 
    quotes = pd.DataFrame()
    for symbol in tickers:    
        df = pd.DataFrame(client.get_historical_klines(symbol, interval, start, end))
        df = df.iloc[:,[0, 4]]
        df.columns = ["Date", symbol]
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index, unit="ms")
        df[symbol] = df[symbol].astype(float)
        quotes = quotes.merge(df, how="outer", left_index=True, right_index=True)

    quotes = quotes.ffill() # fill missing values in a DataFrame or Series by forward filling them.

    return quotes









################################################################################
#                        CORRELATION ANALYSIS                                  #

def correlation_map(df, threshold=None):
    quotes_pct_ret_corr = df.pct_change().dropna().corr()

    if threshold is not None:
        # Apply the threshold: keep only correlations above the threshold
        quotes_pct_ret_corr = quotes_pct_ret_corr.applymap(lambda x: x if abs(x) >= threshold else np.nan)

    plt.subplots(figsize=(10, 8))
    sns.heatmap(quotes_pct_ret_corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
    plt.show()
    

def dispersion_map(df, column1, column2):
    df_pct_ret = df.pct_change().ffill()

    # Fazer scatter plot
    fig = px.scatter(df_pct_ret, x=column1, y=column2, width=900, height=500, trendline="ols", trendline_color_override="black", title='regressão linear OLS')

    fig.update_layout(xaxis=dict(title=f"{column1}"), yaxis = dict(title=f"{column1}"))
    
    # Acrescentar tema
    fig = fig.update_layout(template='seaborn')

    fig.show()

################################################################################
#                          RISK ANALYSIS                                       #

def compute_std(df, frequency="daily"):
    
    # Raise an error if the frequency is not valid
    if frequency not in frequency_map:
        raise ValueError("Please use only: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    # Calculate the percentage change and standard deviation
    df_returns = df.pct_change().dropna()
    return df_returns.std() * np.sqrt(frequency_map[frequency])



def std_graph(df, frequency):
    # Raise an error if the frequency is not valid
    if frequency not in frequency_map:
        raise ValueError("Please use only: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    annual_std_dev = compute_std(df,frequency)

    # Format the annual std devs as percents with two decimal places
    annual_std_dev_pct = annual_std_dev.apply(lambda x: '{:.2%}'.format(x))

    # Create a bar chart
    std_dev = go.Figure(data=[
        go.Bar(x=annual_std_dev, y=annual_std_dev.index, orientation='h',
            text=annual_std_dev_pct, textposition='auto')])
    
    # Customize plot layout
    std_dev.update_layout(
        title='Standard Deviation per Asset Class',
        xaxis_title='Standard Deviation',
        yaxis_title='Asset Class',
        autosize=False,
        width=1200,
        height=700,
        margin=dict(l=100),
        bargap=0.15
    ).show()

def compute_maxDD(df):
    cumulative_returns = (1 + df.pct_change().ffill()).cumprod() # Cumulative Product
    peak = cumulative_returns.cummax() # Cumulative Maximum
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


#############################################################################
#                           RETURN ANALYSIS                                 #


def compute_g(FV, PV, n):
    '''
    g = compounded growth
    FV = Future Value
    PV = Present Value
    n = years
    '''
    g = ((FV/PV)**(1/n))-1
    return g

def compute_CAGR(df, days_per_year="days_per_year"):
    '''
    df = dataframe
    '''
    # CAGR = End Value / Begin Value ^ (1/years) - 1
    return (df.iloc[-1]/df.iloc[0]) ** (1/compute_years(df, days_per_year))-1 # ** = ^

def bar_graph(df):
    # Calcular os retornos anuais
    df_annual_pct_ret = df.pct_change().dropna()
    df_annual_pct_ret.index = df.index[1:]

    year_returns = go.Figure()
    #   Iterate over the tickers list
    for ticker in df.columns:
        year_returns.add_trace(go.Bar(name= f'{ticker}', x=df_annual_pct_ret.index, y=df_annual_pct_ret[ticker]))

    # Alterações ao gráfico
    year_returns.update_layout(barmode='group', # Colocar as barras em grupo
            title='Análise rentabilidades classes de ativos',
            yaxis_tickformat = '.2%', # Colocar o eixo dos y em percentagem
            template='simple_white', # Escolhemos um template que achemos bonito
            height=700, width=1200,
            showlegend=True,
            legend=dict(yanchor="bottom", y=1, xanchor="center", x=0.5, orientation='h')
            ).show()


def comparative_graph(df):
    # Fazer com que as séries temporais comecem do mesmo ponto
    df_normalized = normalize(df)

    performance_compare = go.Figure()
    # Iterate over the tickers list
    for ticker in df.columns:
        performance_compare.add_trace(go.Scatter(name=f'{ticker}', x=df_normalized.index, y=df_normalized[ticker]))

    # Acrescentar tema
    performance_compare.update_layout(template='simple_white',
                   title='Análise performance classes de ativos',
                   height=700,
                   width=1200,
                   hovermode="x unified",
                   showlegend=True,
                   legend=dict(yanchor="bottom", y=1, xanchor="center",x=0.5, orientation='h'),
                   xaxis=dict(showgrid=False, zeroline=False, showline=False, mirror=True),
                   yaxis=dict(title='Valorização (%)', ticksuffix=' %', showgrid=False, zeroline=True, showline=False, mirror=True, type='log'),
                   ).show()


def rolling_mean(df,rolling_nr):
    rolling_df = df.pct_change().dropna()
    return rolling_df.rolling(rolling_nr).mean().dropna()









##############################################################################
#                       RISK-RETURN ANALYSIS                                 #

def compute_sharpe(df, rf, days_per_year, frequency="daily"):
    # Raise an error if the frequency is not valid
    if frequency not in frequency_map:
        raise ValueError("Please use only: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    cagr = compute_CAGR(df, days_per_year)
    std = compute_std(df, frequency)
    return ( cagr - rf ) / std


def sharpe_ratio_graph(df,rf="rf", frequency="daily", days_per_year="days_per_year", treshold=0):
    # Raise an error if the frequency is not valid
    if frequency not in frequency_map:
        raise ValueError("Please use only: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    annualized_returns = compute_CAGR(df, days_per_year)
    annual_std_dev = compute_std(df, frequency)
    sharpe_ratio = (annualized_returns - rf) / annual_std_dev

    # Create scatter plot
    plt.figure(figsize=(12, 10))
    plt.scatter(annualized_returns, annual_std_dev, c=sharpe_ratio, cmap='coolwarm', s=sharpe_ratio*400,
                alpha=0.6, edgecolors='w')#  # sizing points by Sharpe Ratio

    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Annualized Return')
    plt.ylabel('Annual Std Dev')
    plt.title('Asset Class Performance: Sharpe Ratio, Return & Risk')

    # Annotate asset class labels
    for ticker in df.columns:
        if sharpe_ratio[ticker] >= treshold:
            plt.annotate(ticker, (annualized_returns[ticker], annual_std_dev[ticker]))

    plt.show()





#################################################################################
#                       MONTE CARLO SIMULATIONS                                 #

def monte_carlo_simulation(nr_simulations, nr_days_simulation, expected_return, portfolio_std, initial_portfolio_value, days_per_year="days_per_year"):
    # Simulação de Monte Carlo
    simulation_results = np.zeros((nr_days_simulation, nr_simulations))

    for i in range(nr_simulations):
        # Gerando retornos diários assumindo distribuição normal
        daily_return_mean = expected_return / days_per_year
        daily_return_std = portfolio_std / np.sqrt(days_per_year)
        daily_returns = np.random.normal(daily_return_mean, daily_return_std, nr_days_simulation)
        # Calculando o valor acumulado da carteira
        simulation_results[:, i] = initial_portfolio_value * np.cumprod(1 + daily_returns)

    # Converter resultados para DataFrame
    results_df = pd.DataFrame(simulation_results)
    return results_df


def monte_carlo_graph(df):
    # Calcular a média das simulações
    mean_simulation = df.mean(axis=1)

    # Criar uma figura Plotly para as simulações
    MonteCarlo = go.Figure()

    '''
    # Adicionar cada linha ao gráfico
    for col in results_df.columns:
        MonteCarlo.add_trace(go.Scatter(x=results_df.index, y=results_df[col], mode='lines', line=dict(color='grey'), opacity=0.5))
    '''

    # Selecionar 100 linhas aleatorias para plotar
    selected_lines = df.sample(n=100, axis=1, random_state=1)
    selected_lines = pd.concat([selected_lines, df[[df.idxmax(axis=1).iloc[-1]]], df[[df.idxmin(axis=1).iloc[-1]]]], axis=1)  # Adiciona a máxima e mínima

    # Adicionar cada linha selecionada ao gráfico
    for col in selected_lines.columns:
        MonteCarlo.add_trace(go.Scatter(x=selected_lines.index, y=selected_lines[col], mode='lines', line=dict(color='grey'), opacity=0.5))

    # Adicionar a linha média ao gráfico
    MonteCarlo.add_trace(go.Scatter(name='mean', x=df.index, y=mean_simulation, mode='lines', line=dict(color='red', dash='dash')))

    # Alterações ao gráfico
    MonteCarlo.update_layout(
        title='Simulação Monte Carlo',
        yaxis_title='Valor carteira',
        yaxis_ticksuffix=' €',
        xaxis_title="Nr de dias simulados",
        template='simple_white',
        height=650, width=1200,
        showlegend=False,
        legend=dict(yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    ).show()


def monte_carlo_histogram(df):
    final_values = df.iloc[-1]
    expected_final_value = df.mean(axis=1)

    # Criar uma figura Plotly para o histograma
    histogram = go.Figure()
    # Adicionar o histograma ao gráfico
    histogram.add_trace(go.Histogram(x=final_values, nbinsx=30, marker=dict(color='grey', opacity=0.5, line=dict(color='black', width=1))))

    # Adicionar linha da média ao histograma
    histogram.add_trace(go.Scatter(x=[expected_final_value, expected_final_value], y=[0, 10], mode='lines', line=dict(color='red', dash='dash')))

    # Alterações ao gráfico
    histogram.update_layout(
        title=f'Distribuição dos Valores Finais da Carteira',
        xaxis_title='Valor carteira',
        xaxis_ticksuffix=' €',
        yaxis_title='Frequência',
        template='simple_white',
        height=700, width=1200,
        showlegend=False,
    ).show()




#################################################################################
#                             Metrics Table                                     #


def metrics_table(df, rf, days_per_year, frequency="daily"):    
    # Raise an error if the frequency is not valid
    if frequency not in frequency_map:
        raise ValueError("Please use only: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    # Initialize a DataFrame to hold the metrics
    metrics = pd.DataFrame(index = df.columns)

    df_returns = df.pct_change().ffill()

    # Step 1: Calculate CAGR
    annual_returns = compute_CAGR(df, days_per_year)
    metrics["CAGR"] = annual_returns

    # Step 2: Calculate Standard Deviation
    metrics["StdDev"] = compute_std(df, frequency)

    # Step 3: Calculate the Sharpe ratio
    metrics["Sharpe"] = compute_sharpe(df,rf,days_per_year,frequency)

    # Step 3: Calculate the Maximum Drawdown (MDD)
    metrics["Max DD"] = compute_maxDD(df)

    # metrics.index[0] = Mercado / Benchmark

    # Step 4: Calculate Beta
    beta_list = []
    for ticker in df.columns:
        # Calculate covariance of the ticker with the market and variance of the market
        covariance = df_returns[ticker].cov(df_returns.iloc[:, 0]) 
        variance = df_returns.iloc[:, 0].var()
        beta_list.append(covariance / variance) # Calcula e junta o cálculo à lista beta
    metrics["Beta"] = beta_list

    # Step 5: Calculate the Alpha (assuming CAPM)
    # Alpha = Asset Return - CAPM
    alpha_list = []
    for ticker in df.columns:
        if ticker != metrics.index[0]:
            alpha_calculo = annual_returns[ticker] - (rf + metrics.at[ticker, "Beta"] * (annual_returns[0] - rf))
        else:
            alpha_calculo = 0
        alpha_list.append(alpha_calculo)
    metrics["Alpha"] = alpha_list

    return metrics



