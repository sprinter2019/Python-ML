""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""Market indicators  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			 		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  					  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  				  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data  	
import matplotlib.pyplot as plt	  
	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# Simple Moving Average  	

def get_market_indicators(df, sym):
    df_prices = df[sym]
    df_prices = df_prices/df_prices[0]
    market_indicators = pd.DataFrame(index=df_prices.index)
    market_indicators['prices'] = df_prices
    
    # Detremine Simple Moving Average
    sma = df_prices.rolling(window =14, center=False).mean()
    market_indicators['sma'] = sma
    market_indicators['price/sma'] = df_prices/sma

    # Detremine Boolinger Band Parameters
    stdev = df_prices.rolling(window =14, center=False).std()
    market_indicators['upper band'] = sma + 2*stdev
    market_indicators['lower band'] = sma - 2*stdev
    market_indicators['bollinger value'] = (df_prices-sma)/(2*stdev)

    #Determine Volatility
    market_indicators['volatility'] = stdev


    #Detremine Momentum
    shift_window = 14
    market_indicators['momentum'] = df_prices/df_prices.shift(periods = shift_window)-1

    #Commodity Channel Index
    cci_window = 14
    avg_prices = df_prices.rolling(window =cci_window, center=False).mean()
    market_indicators['cc index'] = (df_prices-avg_prices)/(df_prices.std())

    return market_indicators


def author():
    return "mhassan49"

def test_code():  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    symbol='JPM'
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates).ffill().bfill()
    market_indicators = get_market_indicators(prices_all, symbol)
    xDates = prices_all.index

    #SMA
    market_indicators[['prices', 'sma', 'price/sma']].plot(figsize=(10, 7))
    title= "Indicator: Simple Moving Average"
    xlabel="Dates"
    ylabel="Normalized Share Price"  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig("sma.png") 
    #plt.show()

    #BBands
    market_indicators[['prices', 'upper band', 'lower band', 'sma']].plot(figsize=(10, 7))
    title= "Indicator: Bollinger Bands"
    xlabel="Dates"
    ylabel="Normalized Share Price"     
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig("bband.png") 
    #plt.show()

    market_indicators[['bollinger value']].plot(figsize=(10, 7))
    title= "Indicator: Bollinger Value"
    xlabel="Dates"
    ylabel="Bollinger Value"     
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig("bvalue.png") 
    #plt.show()

    #Momentum
    fig, ax1 = plt.subplots(figsize=(10,7))
    title= "Indicator: Momentum"
    xlabel="Dates"
    ylabel1="Normalized Share Price"   
    ylabel2="Momentum"    
    line1= ax1.plot(xDates, market_indicators['prices'], color = 'blue', label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax2=ax1.twinx()
    line2= ax2.plot(xDates, market_indicators['momentum'], color = 'red', label=ylabel2)
    ax2.set_ylabel(ylabel2)
    lns = line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    fig.savefig("momentum.png") 

    #CCI
    fig, ax1 = plt.subplots(figsize=(10,7))
    title= "Indicator: Commodity Channel Index"
    xlabel="Dates"
    ylabel1="Normalized Share Price"   
    ylabel2="Commodity Channel Index"    
    line1= ax1.plot(xDates, market_indicators['prices'], color = 'blue', label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax2=ax1.twinx()
    line2= ax2.plot(xDates, market_indicators['cc index'], color = 'red', label=ylabel2)
    ax2.set_ylabel(ylabel2)
    lns = line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    fig.savefig("cci.png") 

    #Volatility
    fig, ax1 = plt.subplots(figsize=(10,7))
    title= "Indicator: Volatility"
    xlabel="Dates"
    ylabel1="Normalized Share Price"   
    ylabel2="Volatility"    
    line1= ax1.plot(xDates, market_indicators['prices'], color = 'blue', label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax2=ax1.twinx()
    line2= ax2.plot(xDates, market_indicators['volatility'], color = 'red', label=ylabel2)
    ax2.set_ylabel(ylabel2)
    lns = line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    fig.savefig("volatility.png") 
    #plt.show()
    	  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
