""""""
""" Benchmark trading		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			 		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  					  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals, port_params
from indicators import get_market_indicators as mIndicators

#Manually optimal strategy
def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    df_prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
    #df_prices = df_prices/df_prices.iloc[0,] # Normalize prices
    dates = pd.date_range(sd, ed)
    syms=[symbol]
    df_prices=df_prices[syms]
    #print(df_prices)
    market_indicators =mIndicators(df_prices, symbol)
    df_orders, buy, sell = execute_actions(df_prices, market_indicators, symbol)

    return df_orders, buy, sell

def execute_actions(df_prices, market_indicators, symbol):
    pos_val = 0
    max_holding = 1000
    min_holding = -1000
    max_trade = 2000
    min_trade = -2000
    shares = []
    dates = []
    buy = []
    sell = []

    prices = market_indicators['prices']
    sma= market_indicators['sma']
    pSMA= market_indicators['price/sma']    # Use Price/SMA
    uBand = market_indicators['upper band']
    lBand = market_indicators['lower band']
    bb = market_indicators['bollinger value']
    bbp = (prices-lBand)/(uBand-lBand)  # Calculate Bolinager Percentage
    momentum = market_indicators['momentum'] # Use Meonetum
    #volatility = market_indicators['volatility']
    #ccI = market_indicators['cc index']
    #print(momentum)
    # dates = df_orders.index
    #print(len(df_prices.index))
    for index in range(len(df_prices.index)):
        if (bbp[index]<0.0 and pSMA[index]<1 and momentum[index]<0) and pos_val == 0:
            #print("Condition1 statisfied")
            pos_val = pos_val + max_holding
            shares.append(max_holding)
            dates.append(df_prices.index[index])
            buy.append(df_prices.index[index])

        elif (bbp[index]<0.0 and pSMA[index]<1 and momentum[index]<0)  and pos_val == min_holding:
            #print("Condition2 statisfied")
            pos_val = pos_val + max_trade
            shares.append(max_trade)
            dates.append(df_prices.index[index])
            buy.append(df_prices.index[index])

        elif (bbp[index]>0 and pSMA[index]>1  and momentum[index]>0) and pos_val == 0:
            #print("Condition3 statisfied")
            pos_val = pos_val + min_holding
            shares.append(min_holding)
            dates.append(df_prices.index[index])
            sell.append(df_prices.index[index])

        elif (bbp[index]>0 and pSMA[index]>1  and momentum[index]>0) and pos_val == max_holding:
            #print("Condition4 statisfied")
            pos_val = pos_val + min_trade
            shares.append(min_trade)
            dates.append(df_prices.index[index])
            sell.append(df_prices.index[index])
        
    #print(len(shares))
    #print(shares)

    if pos_val != 0:
        shares.append(-pos_val)
        dates.append(df_prices.index[len(df_prices.index) - 1])

    df_trades = pd.DataFrame(data=shares, index=dates, columns=['orders'])
    

    return df_trades, buy, sell


def benchmark_data(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    df_trades = get_data([symbol], dates)
    range_shares = [1000, -1000]
    date_data = [df_trades.index[0], df_trades.index[len(df_trades.index) - 1]]
    df_orders = pd.DataFrame(data=range_shares, index=date_data, columns=['orders'])

    return df_orders


def author():
    return "mhassan49"


def test_code():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Helper function to test code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    #In-sample trading 
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbols = 'JPM'
    #symbols = 'ML4T-220'
    #symbols = 'AAPL'
    #symbols = 'SINE_FAST_NOISE'
    #symbols = 'UNH'
    sv=100000
    impact = 0.005
    commission = 9.95

    df_trades, long_pos, short_pos = testPolicy(symbols, start_date, end_date, sv)
    #print(df_trades)
    port_vals = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
    df_Benchmark = benchmark_data(symbols, start_date, end_date, sv)
    port_vals_Benchmark = compute_portvals(df_Benchmark, start_val=sv, commission=commission, impact=impact)

    port_vals = port_vals[0]
    port_vals_Benchmark = port_vals_Benchmark[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_params(port_vals)
    cum_ret_Benchmark, avg_daily_ret_Benchmark, std_daily_ret_Benchmark, sharpe_ratio_Benchmark = port_params(port_vals_Benchmark)

    # Compare portfolio against Benchmark
    print(f"Date Range (In-sample): {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Manual (In-sample): {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark (In-sample): {sharpe_ratio_Benchmark}")
    print()
    print(f"Cumulative Return of Manual (In-sample): {cum_ret}")
    print(f"Cumulative Return of Benchmark (In-sample): {cum_ret_Benchmark}")
    print()
    print(f"Standard Deviation of Manual (In-sample): {std_daily_ret}")
    print(f"Standard Deviation of Benchmark (In-sample): {std_daily_ret_Benchmark}")
    print()
    print(f"Average Daily Return of Manual (In-sample): {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark (In-sample): {avg_daily_ret_Benchmark}")
    print()
    print(f"Final Portfolio Value of Manual (In-sample): {port_vals[-1]}")
    print(f"Final Portfolio Value of Benchmark (In-sample): {port_vals_Benchmark[-1]}")
    
    port_vals_Benchmark_norm = port_vals_Benchmark / port_vals_Benchmark.iloc[0,]
    port_vals_norm = port_vals / port_vals.iloc[0,]
    port_vals_Benchmark_norm = port_vals_Benchmark_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()

    df_temp = pd.concat(
        [port_vals_norm, port_vals_Benchmark_norm], keys=["Portfolio", "Benchmark"], axis=1
    )
    df_temp.columns = ['Manual Strategy', 'Benchmark']
    """Plot stock prices with a custom title and meaningful axis labels."""
    title = "Manual Strategy vs Benchmark: In-sample Performance"
    xlabel = "Date"
    ylabel = "Normalized value"
    ax = df_temp.plot(title=title, color = ["red","green"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    plt.vlines(long_pos,ymin,ymax,color='blue')
    plt.vlines(short_pos,ymin,ymax,color='black')
    plt.tight_layout()
    plt.savefig('mos_in.png')
    #plt.show()

    
    # #Outsample trading
    start_date = '2010-1-1'
    end_date = '2011-12-31'
    #symbols = 'JPM'
    #symbols = 'ML4T-220'
    #symbols = 'AAPL'
    #symbols = 'SINE_FAST_NOISE'
    #symbols = 'UNH'
    #sv=100000


    df_trades, long_pos, short_pos = testPolicy(symbols, start_date, end_date, sv)
    port_vals = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
    df_Benchmark = benchmark_data(symbols, start_date, end_date, sv)
    port_vals_Benchmark = compute_portvals(df_Benchmark, start_val=sv, commission=commission, impact=impact)

    port_vals = port_vals[0]
    port_vals_Benchmark = port_vals_Benchmark[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_params(port_vals)
    cum_ret_Benchmark, avg_daily_ret_Benchmark, std_daily_ret_Benchmark, sharpe_ratio_Benchmark = port_params(port_vals_Benchmark)

    # Compare portfolio against Benchmark
    print(f"Date Range (Out-of-sample): {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Manual (Out-of-sample): {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark (Out-of-sample): {sharpe_ratio_Benchmark}")
    print()
    print(f"Cumulative Return of Manual (Out-of-sample): {cum_ret}")
    print(f"Cumulative Return of Benchmark (Out-of-sample): {cum_ret_Benchmark}")
    print()
    print(f"Standard Deviation of Manual (Out-of-sample): {std_daily_ret}")
    print(f"Standard Deviation of Benchmark (Out-of-sample): {std_daily_ret_Benchmark}")
    print()
    print(f"Average Daily Return of Manual (Out-of-sample): {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark (Out-of-sample): {avg_daily_ret_Benchmark}")
    print()
    print(f"Final Portfolio Value of Manual (Out-of-sample): {port_vals[-1]}")
    print(f"Final Portfolio Value of Benchmark (Out-of-sample): {port_vals_Benchmark[-1]}")
    
    port_vals_Benchmark_norm = port_vals_Benchmark / port_vals_Benchmark.iloc[0,]
    port_vals_norm = port_vals / port_vals.iloc[0,]
    port_vals_Benchmark_norm = port_vals_Benchmark_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()

    df_temp = pd.concat(
        [port_vals_norm, port_vals_Benchmark_norm], keys=["Portfolio", "Benchmark"], axis=1
    )
    df_temp.columns = ['Manual Strategy', 'Benchmark']
    """Plot stock prices with a custom title and meaningful axis labels."""
    title = "Manual Strategy vs Benchmark: Out-of-sample Performance"
    xlabel = "Date"
    ylabel = "Normalized value"
    ax = df_temp.plot(title=title, color = ["red","green"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.vlines(long_pos,ymin,ymax,color='blue')
    plt.vlines(short_pos,ymin,ymax,color='black')
    plt.tight_layout()
    plt.savefig('mos_out.png')
    #plt.show()

if __name__ == "__main__":
    test_code()
