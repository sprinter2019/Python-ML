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
import StrategyLearner as sl
import ManualStrategy as ms
import random

def author():
    return "mhassan49"

def test_code():
    #random.seed(903450999)
    #np.random.seed(903450999)
    random.seed(1481090000)
    np.random.seed(1481090000)
    
    #Insample
    start_date_in = '2008-1-1'
    end_date_in = '2009-12-31'
    # #Outsample
    start_date_out = '2010-1-1'
    end_date_out = '2011-12-31'
    symbols = 'JPM'
    #symbols = 'ML4T-220'
    #symbols = 'ML4T-050'
    #symbols = 'ML4T-399'
    #symbols = 'ZMH'
    #symbols = 'NVDA'
    #symbols = 'AAPL'
    #symbols = 'SINE_FAST_NOISE'
    #symbols = 'UNH'
    sv=100000
    impact = 0.005
    commission = 9.95

    df_trades,buy,sell = ms.testPolicy(symbols, start_date_in, end_date_in, sv)
    #print(df_trades)
    port_vals = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
    df_Benchmark = ms.benchmark_data(symbols, start_date_in, end_date_in, sv)
    port_vals_Benchmark = compute_portvals(df_Benchmark, start_val=sv, commission=commission, impact=impact)

    port_vals = port_vals[0]
    port_vals_Benchmark = port_vals_Benchmark[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_params(port_vals)
    cum_ret_Benchmark, avg_daily_ret_Benchmark, std_daily_ret_Benchmark, sharpe_ratio_Benchmark = port_params(port_vals_Benchmark)

    sLearner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    sLearner.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades = sLearner.testPolicy(symbols, start_date_in, end_date_in, sv)
    #print(df_trades)
    #print(sl_trades)
    port_vals_strategy = compute_portvals(sl_trades, start_val=sv, commission=commission, impact=impact)
    port_vals_strategy = port_vals_strategy[0]
    cum_ret_strategy, avg_daily_ret_strategy, std_daily_ret_strategy, sharpe_ratio_strategy = port_params(port_vals_strategy)
    port_vals_strategy_norm = port_vals_strategy / port_vals_strategy.iloc[0,]
    port_vals_strategy_norm = port_vals_strategy_norm.to_frame()

   # Compare portfolio against Benchmark
    print(f"Date Range (In-sample): {start_date_in} to {end_date_in}")
    print()
    print(f"Sharpe Ratio of Manual (In-sample): {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark (In-sample): {sharpe_ratio_Benchmark}")
    print(f"Sharpe Ratio of Strategy (In-sample): {sharpe_ratio_strategy}")
    print()
    print(f"Cumulative Return of Manual (In-sample): {cum_ret}")
    print(f"Cumulative Return of Benchmark (In-sample): {cum_ret_Benchmark}")
    print(f"Cumulative Return of Strategy(In-sample): {cum_ret_strategy}")
    print()
    print(f"Standard Deviation of Manual (In-sample): {std_daily_ret}")
    print(f"Standard Deviation of Benchmark (In-sample): {std_daily_ret_Benchmark}")
    print(f"Standard Deviation of Strategy(In-sample): {std_daily_ret_strategy}")
    print()
    print(f"Average Daily Return of Manual (In-sample): {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark (In-sample): {avg_daily_ret_Benchmark}")
    print(f"Average Daily Return of Strategy(In-sample): {avg_daily_ret_strategy}")
    print()
    print(f"Final Portfolio Value of Manual (In-sample): {port_vals[-1]}")
    print(f"Final Portfolio Value of Benchmark (In-sample): {port_vals_Benchmark[-1]}")
    print(f"Final Portfolio Value of Strategy(In-sample): {port_vals_strategy[-1]}")
    
    
    port_vals_Benchmark_norm = port_vals_Benchmark / port_vals_Benchmark.iloc[0,]
    port_vals_norm = port_vals / port_vals.iloc[0,]
    port_vals_Benchmark_norm = port_vals_Benchmark_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()

    df_temp = pd.concat(
        [port_vals_strategy_norm, port_vals_norm, port_vals_Benchmark_norm], keys=["Strategy", "Manual", "Benchmark"], axis=1
    )
    df_temp.columns = ["Strategy Learner", 'Manual Strategy', 'Benchmark']
    """Plot stock prices with a custom title and meaningful axis labels."""
    title = "Strategy Learner vs Manual Strategy vs Benchmark: In-sample Performance "
    xlabel = "Date"
    ylabel = "Normalized value"
    ax = df_temp.plot(title=title, color = ["red","green","blue"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig('sos_in.png')
    ##plt.show()


    df_trades, buy, sell = ms.testPolicy(symbols, start_date_out, end_date_out, sv)
    #print(df_trades)
    port_vals = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
    df_Benchmark = ms.benchmark_data(symbols, start_date_out, end_date_out, sv)
    port_vals_Benchmark = compute_portvals(df_Benchmark, start_val=sv, commission=commission, impact=impact)

    port_vals = port_vals[0]
    port_vals_Benchmark = port_vals_Benchmark[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_params(port_vals)
    cum_ret_Benchmark, avg_daily_ret_Benchmark, std_daily_ret_Benchmark, sharpe_ratio_Benchmark = port_params(port_vals_Benchmark)

    sLearner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    sLearner.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades = sLearner.testPolicy(symbols, start_date_out, end_date_out, sv)
    #print(df_trades)
    #print(sl_trades)
    port_vals_strategy = compute_portvals(sl_trades, start_val=sv, commission=commission, impact=impact)
    port_vals_strategy = port_vals_strategy[0]
    cum_ret_strategy, avg_daily_ret_strategy, std_daily_ret_strategy, sharpe_ratio_strategy = port_params(port_vals_strategy)
    port_vals_strategy_norm = port_vals_strategy / port_vals_strategy.iloc[0,]
    port_vals_strategy_norm = port_vals_strategy_norm.to_frame()

   # Compare portfolio against Benchmark
    print(f"Date Range (Out-sample): {start_date_out} to {end_date_out}")
    print()
    print(f"Sharpe Ratio of Manual (Out-sample): {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark (Out-sample): {sharpe_ratio_Benchmark}")
    print(f"Sharpe Ratio of Strategy (Out-sample): {sharpe_ratio_strategy}")
    print()
    print(f"Cumulative Return of Manual (Out-sample): {cum_ret}")
    print(f"Cumulative Return of Benchmark (Out-sample): {cum_ret_Benchmark}")
    print(f"Cumulative Return of Strategy(Out-sample): {cum_ret_strategy}")
    print()
    print(f"Standard Deviation of Manual (Out-sample): {std_daily_ret}")
    print(f"Standard Deviation of Benchmark (Out-sample): {std_daily_ret_Benchmark}")
    print(f"Standard Deviation of Strategy(Out-sample): {std_daily_ret_strategy}")
    print()
    print(f"Average Daily Return of Manual (Out-sample): {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark (Out-sample): {avg_daily_ret_Benchmark}")
    print(f"Average Daily Return of Strategy(Out-sample): {avg_daily_ret_strategy}")
    print()
    print(f"Final Portfolio Value of Manual (Out-sample): {port_vals[-1]}")
    print(f"Final Portfolio Value of Benchmark (Out-sample): {port_vals_Benchmark[-1]}")
    print(f"Final Portfolio Value of Strategy(Out-sample): {port_vals_strategy[-1]}")
    
    
    port_vals_Benchmark_norm = port_vals_Benchmark / port_vals_Benchmark.iloc[0,]
    port_vals_norm = port_vals / port_vals.iloc[0,]
    port_vals_Benchmark_norm = port_vals_Benchmark_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()

    df_temp = pd.concat(
        [port_vals_strategy_norm, port_vals_norm, port_vals_Benchmark_norm], keys=["Strategy", "Manual", "Benchmark"], axis=1
    )
    df_temp.columns = ["Strategy Learner", 'Manual Strategy', 'Benchmark']
    """Plot stock prices with a custom title and meaningful axis labels."""
    title = "Strategy Learner vs Manual Strategy vs Benchmark: Out-of-sample Performance "
    xlabel = "Date"
    ylabel = "Normalized value"
    ax = df_temp.plot(title=title, color = ["red","green","blue"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig('sos_out.png')
    #plt.show()

if __name__ == "__main__":
    test_code()