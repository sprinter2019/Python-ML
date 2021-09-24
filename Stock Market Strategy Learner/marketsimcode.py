""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import os  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def compute_portvals(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    data,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    start_val=1000000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    commission=0.0,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    impact=0.0,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Computes the portfolio values.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param orders_file: Path of the order file or the file object  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type orders_file: str or file object  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type start_val: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type commission: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type impact: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this is the function the autograder will call to test your code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # code should work correctly with either input  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # TODO: Your code here  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # In the template, instead of computing the value of the portfolio, we just  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # read in the value of IBM over 6 months  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #start_date = dt.datetime(2008, 1, 1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #end_date = dt.datetime(2008, 6, 1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #portvals = get_data(["IBM"], pd.date_range(start_date, end_date))  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #portvals = portvals[["IBM"]]  # remove SPY  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #rv = pd.DataFrame(index=portvals.index, data=portvals.values)  
     
    #data = pd.read_csv (orders_file, index_col='Date', parse_dates=True, na_values='nan')	
    data = updateSymbols(data)
    start_date = data.index.min()
    end_date = data.index.max()	  	  
    stock_syms = data.Symbol.unique().tolist()	
    #print(data)
    #print(start_date)
    #print(end_date)
    prices = get_data(stock_syms, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0
    prices.ffill().bfill()	
    
    trades = exceute_trades(data, prices, start_val, stock_syms, commission, impact) 
    holdings = update_holdings(trades, start_val)	

    value = prices*holdings
    port_val = pd.DataFrame(value.sum(axis=1), value.index)

    return port_val		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    

def exceute_trades(data, prices, start_val, stock_syms, commission, impact):
    trades = pd.DataFrame(np.zeros((prices.shape)), prices.index, prices.columns)
    for i, r in data.iterrows():
        row = r['Symbol']
        total_value = prices.loc[i, row] * r['Shares']
        trading_cost = impact*total_value+commission

        if r['Order'] == 'BUY':
            trades.loc[i, row] = trades.loc[i, row] + r['Shares']
            trades.loc[i, 'Cash'] = trades.loc[i, 'Cash'] - total_value - trading_cost
        if r['Order'] == 'SELL':
            trades.loc[i, row] = trades.loc[i, row] - r['Shares']
            trades.loc[i, 'Cash'] = trades.loc[i, 'Cash'] + total_value - trading_cost
    return trades

def update_holdings(trades, start_val):
    holdings = pd.DataFrame(np.zeros((trades.shape)), trades.index, trades.columns)
    for iter in range(len(holdings)):
        if iter == 0:
            holdings.iloc[0,:-1] = trades.iloc[0,:-1]. copy()
            holdings.iloc[0,-1] = trades.iloc[0,-1] + start_val
        else:
            holdings.iloc[iter] = holdings.iloc[iter-1]+trades.iloc[iter]

    return holdings

def updateSymbols(df):
    symbol = []
    order = []
    share = []
    for i in range(len(df.index)):
        symbol.append('JPM')
        if df['orders'][i] > 0:
            order.append('BUY')
            share.append(df['orders'][i])
        elif df['orders'][i] < 0:
            order.append('SELL')	
            share.append(-df['orders'][i])
    df_symbol = pd.DataFrame(data = symbol, index = df.index, columns = ['Symbol'])			
    df_order = pd.DataFrame(data = order, index = df.index, columns = ['Order'])	
    df_share = pd.DataFrame(data = share, index = df.index, columns = ['Shares'])	
    
    df_updatedSym = df_symbol.join(df_order).join(df_share)
	#print df_result
    return df_updatedSym

# Following mdethod is adopted from project 2  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def port_params(port_val):
    cr = (port_val[-1] - port_val[0]) / port_val[0]
    daily_ret=(port_val/port_val.shift(1))-1
    risk_free_ret=0 
    adr = np.mean(daily_ret)
    sddr = np.std(daily_ret)
    sr = np.mean(daily_ret - risk_free_ret) / np.std(daily_ret - risk_free_ret)
    sr = sr * np.sqrt(252)
    return cr, adr, sddr, sr

def author():
    return "mhassan49"

def test_code():  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Helper function to test code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this is a helper function you can use to test your code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # note that during autograding his function will not be called.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Define input parameters  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    of = "./orders/orders2.csv"  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    sv = 1000000  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Process orders  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    if isinstance(portvals, pd.DataFrame):  	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    else:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        "warning, code did not return a DataFrame"  

    #print(author())	
    ##print(portvals)	
      	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Get portfolio stats  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    '''
    start_date = dt.datetime(2008, 1, 1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    end_date = dt.datetime(2008, 6, 1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.2,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.01,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.02,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        1.5,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.2,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.01,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        0.02,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        1.5,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ]  	
    '''

    df = pd.read_csv(of, index_col = 'Date', parse_dates = True, na_values = ['nan'])
    start_date = min(df.index)
    end_date = max(df.index)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_params(portvals)
  
    portvals_SPY = get_data(['SPY'], dates = pd.date_range(start_date, end_date))
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_params(portvals_SPY[portvals_SPY.columns[0]])
    	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			


    # Compare portfolio against $SPX  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
