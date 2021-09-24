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
    random.seed(1481090000)
    np.random.seed(1481090000)
    #Insample
    start_date_in = '2008-1-1'
    end_date_in = '2009-12-31'
    # #Outsample
    start_date_out = '2010-1-1'
    end_date_out = '2011-12-31'
    symbols = 'JPM'
    sv=100000

    impact1=0.0
    impact2=0.005
    impact3=0.010
    impact4=0.015

    ##Impact 1
    sLearner1 = sl.StrategyLearner(verbose=False, impact=impact1, commission=0.0)
    sLearner1.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades1 = sLearner1.testPolicy(symbols, start_date_in, end_date_in, sv)
   
    port_vals_strategy1 = compute_portvals(sl_trades1, start_val=sv, commission=0.00, impact=impact1)
    port_vals_strategy1 = port_vals_strategy1[0]
    cum_ret_strategy1, avg_daily_ret_strategy1, std_daily_ret_strategy1, sharpe_ratio_strategy1 = port_params(port_vals_strategy1)
    port_vals_strategy_norm1 = port_vals_strategy1 / port_vals_strategy1.iloc[0,]
    port_vals_strategy_norm1 = port_vals_strategy_norm1.to_frame()

    ##Impact 2
    sLearner2 = sl.StrategyLearner(verbose=False, impact=impact2, commission=0.0)
    sLearner2.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades2 = sLearner2.testPolicy(symbols, start_date_in, end_date_in, sv)
   
    port_vals_strategy2 = compute_portvals(sl_trades2, start_val=sv, commission=0.00, impact=impact2)
    port_vals_strategy2 = port_vals_strategy2[0]
    cum_ret_strategy2, avg_daily_ret_strategy2, std_daily_ret_strategy2, sharpe_ratio_strategy2 = port_params(port_vals_strategy2)
    port_vals_strategy_norm2 = port_vals_strategy2 / port_vals_strategy2.iloc[0,]
    port_vals_strategy_norm2 = port_vals_strategy_norm2.to_frame()

    ##Impact 3
    sLearner3 = sl.StrategyLearner(verbose=False, impact=impact3, commission=0.0)
    sLearner3.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades3 = sLearner3.testPolicy(symbols, start_date_in, end_date_in, sv)
   
    port_vals_strategy3 = compute_portvals(sl_trades3, start_val=sv, commission=0.00, impact=impact3)
    port_vals_strategy3 = port_vals_strategy3[0]
    cum_ret_strategy3, avg_daily_ret_strategy3, std_daily_ret_strategy3, sharpe_ratio_strategy3 = port_params(port_vals_strategy3)
    port_vals_strategy_norm3 = port_vals_strategy3 / port_vals_strategy3.iloc[0,]
    port_vals_strategy_norm3 = port_vals_strategy_norm3.to_frame()

    ##Impact 4
    sLearner4 = sl.StrategyLearner(verbose=False, impact=impact4, commission=0.0)
    sLearner4.add_evidence(symbol=symbols, sd=start_date_in, ed=end_date_in, sv=sv)
    sl_trades4 = sLearner4.testPolicy(symbols, start_date_in, end_date_in, sv)
   
    port_vals_strategy4 = compute_portvals(sl_trades4, start_val=sv, commission=0.00, impact=impact4)
    port_vals_strategy4 = port_vals_strategy4[0]
    cum_ret_strategy4, avg_daily_ret_strategy4, std_daily_ret_strategy4, sharpe_ratio_strategy4 = port_params(port_vals_strategy4)
    port_vals_strategy_norm4 = port_vals_strategy4 / port_vals_strategy4.iloc[0,]
    port_vals_strategy_norm4 = port_vals_strategy_norm4.to_frame()

   # Compare portfolio against Benchmark
    print(f"Date Range (In-sample): {start_date_in} to {end_date_in}")
    print()
    print(f"Sharpe Ratio of Strategy (Impact 1): {sharpe_ratio_strategy1}")
    print(f"Sharpe Ratio of Strategy (Impact 2): {sharpe_ratio_strategy2}")
    print(f"Sharpe Ratio of Strategy (Impact 3): {sharpe_ratio_strategy3}")
    print(f"Sharpe Ratio of Strategy (Impact 4): {sharpe_ratio_strategy4}")
    print()
    print(f"Cumulative Return of Strategy(Impact 1): {cum_ret_strategy1}")
    print(f"Cumulative Return of Strategy(Impact 2): {cum_ret_strategy2}")
    print(f"Cumulative Return of Strategy(Impact 3): {cum_ret_strategy3}")
    print(f"Cumulative Return of Strategy(Impact 4): {cum_ret_strategy4}")
    print()
    print(f"Standard Deviation of Strategy(Impact 1): {std_daily_ret_strategy1}")
    print(f"Standard Deviation of Strategy(Impact 2): {std_daily_ret_strategy2}")
    print(f"Standard Deviation of Strategy(Impact 3: {std_daily_ret_strategy3}")
    print(f"Standard Deviation of Strategy(Impact 4): {std_daily_ret_strategy4}")
    print()
    print(f"Average Daily Return of Strategy(Impact 1): {avg_daily_ret_strategy1}")
    print(f"Average Daily Return of Strategy(Impact 2): {avg_daily_ret_strategy2}")
    print(f"Average Daily Return of Strategy(Impact 3): {avg_daily_ret_strategy3}")
    print(f"Average Daily Return of Strategy(Impact 4): {avg_daily_ret_strategy4}")
    print()
    print(f"Final Portfolio Value of Strategy(Impact 1): {port_vals_strategy1[-1]}")
    print(f"Final Portfolio Value of Strategy(Impact 2): {port_vals_strategy2[-1]}")
    print(f"Final Portfolio Value of Strategy(Impact 3: {port_vals_strategy3[-1]}")
    print(f"Final Portfolio Value of Strategy(Impact 4: {port_vals_strategy4[-1]}")

    df_temp = pd.concat(
        [port_vals_strategy_norm1, port_vals_strategy_norm2, port_vals_strategy_norm3, port_vals_strategy_norm4], keys=["impact1", "impact2", "impact3", "impact4"], axis=1
    )
    df_temp.columns = ["Impact = 0.00", 'Impact = 0.005', 'Impact = 0.010', 'Impact = 0.015']
    """Plot stock prices with a custom title and meaningful axis labels."""
    title = "Effect of Impact parameter on the Strategy Learner "
    xlabel = "Date"
    ylabel = "Normalized value"
    ax = df_temp.plot(title=title, color = ["red","green","blue", "black"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig('impact_in.png')
    #plt.show()


if __name__ == "__main__":
    test_code() 
