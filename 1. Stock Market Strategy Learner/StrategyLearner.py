""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
import random  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  	
import numpy as np	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
#import util as ut  		 

import RTLearner as rt
import BagLearner as bl
from marketsimcode import compute_portvals, port_params
from indicators import get_market_indicators as mIndicators
from util import get_data, plot_data
import matplotlib.pyplot as plt
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
class StrategyLearner(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type impact: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type commission: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.verbose = verbose  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.impact = impact  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.commission = commission  
        #self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)	

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this method should create a QLearner, and train it for trading  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def add_evidence(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=10000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        # add your code to do learning here  	
        Nday=2
        #YBUY = 0.04+0.000+0.065
        #YBUY = 0.04+0.000+0.07
        #YSELL =  -0.01-0.0017
        #YSELL =  -0.01-0.055
        #YBUY = 0.01
        #YSELL =  -0.01
        #YBUY = 0.105
        #YSELL =  -0.0117
        YBUY = 0.0000
        YSELL = -0.0000
        bags= 50
        leaf = 5
        window=14
        	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        df_prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        #df_prices = df_prices/df_prices.iloc[0,] # Normalize prices
        dates = pd.date_range(sd, ed)
        market_indicators =mIndicators(df_prices, symbol)
        syms=[symbol]
        df_prices=df_prices[syms]

        #print(df_prices)

        pos_val = 0
        max_holding = 1000
        min_holding = -1000
        max_trade = 2000
        min_trade = -2000
        shares = []
        dates = []

        prices = market_indicators['prices']
        #print(prices)
        sma= market_indicators['sma']
        #print(sma[window-2:])
        pSMA= market_indicators['price/sma']    # Use Price/SMA
        #pSMA = pSMA[window-1:]
        #print(pSMA[window-2:])
        uBand = market_indicators['upper band']
        lBand = market_indicators['lower band']
        bb = market_indicators['bollinger value']
        bbp = (prices-lBand)/(uBand-lBand)  # Calculate Bolinager Percentage
        #bbp = bbp[window-1:]
        momentum = market_indicators['momentum'] # Use Meonetum
        #momentum = momentum[window-1:]
        #volatility = market_indicators['volatility']
        #ccI = market_indicators['cc index']

        #listIndicators = pd.concat((df_psma, df_bbp, df_momentum), axis=1)
        listIndicators = pd.concat([pSMA, bbp, momentum], axis=1)
        #print(listIndicators)
        #print(len(listIndicators))
        #listIndicators.dropna()
        listIndicators.fillna(0,inplace=True)
        listIndicators = listIndicators[:-Nday]    
        #print(listIndicators)
        #print(len(listIndicators))
        train_x = listIndicators.values

        train_y=[]
        for i in range(df_prices.shape[0]-Nday):
            ret = (df_prices.iloc[i+Nday]-df_prices.iloc[i])/df_prices.iloc[i]
            #print(ret[symbol])
            if ret[symbol] > (YBUY + self.impact):
                train_y.append(1)
            elif ret[symbol] < (YSELL - self.impact):
                train_y.append(-1)
            else:
                train_y.append(0)

        #print(train_y)
        train_y=np.array(train_y)

        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":leaf}, bags = bags, boost = False, verbose = False)
        self.learner.add_evidence(train_x, train_y)


    def author(self):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: The GT username of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        return "mhassan49"  # replace tb34 with your Georgia Tech username  		 	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def testPolicy(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=10000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        

        df_prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        syms=[symbol]
        df_prices=df_prices[syms]
        #print(df_prices)
        #df_prices = df_prices/df_prices.iloc[0,] # Normalize prices
        dates = pd.date_range(sd, ed)
        market_indicators =mIndicators(df_prices, symbol)
        prices = market_indicators['prices']
        sma= market_indicators['sma']
        pSMA= market_indicators['price/sma']    # Use Price/SMA
        uBand = market_indicators['upper band']
        lBand = market_indicators['lower band']
        bb = market_indicators['bollinger value']
        bbp = (prices-lBand)/(uBand-lBand)  # Calculate Bolinager Percentage
        momentum = market_indicators['momentum'] # Use Meonetum
        volatility = market_indicators['volatility']
        ccI = market_indicators['cc index']

        #df_psma = pSMA.rename(columns={symbol:'psma'})
        #df_bbp = bbp.rename(columns={symbol:'bbp'})
        #df_momentum = momentum.rename(columns={symbol:'momentum'})
        listIndicators = pd.concat([pSMA, bbp, momentum], axis=1)
        #listIndicators.dropna()
        listIndicators.fillna(0,inplace=True)
        test_x = listIndicators.values
        test_y = self.learner.query(test_x)
        test_y=test_y.flatten()

        #print(test_x)
        #print(test_y)
    

        #trades = df_prices.copy()
        #trades.loc[:]=0

        #print("SL: ")
        #print(len(df_prices.index))
        pos_val = 0
        max_holding = 1000
        min_holding = -1000
        max_trade = 2000
        min_trade = -2000
        shares = []
        dates = []
        buyAndSell = []

        #(len(test_y))
        #print(test_y)
        temp_holding_total=0
        temp_holding =[]
        temp_dates = []
    
        for index in range(len(df_prices.index)):
            #print(df_prices.index[index])
            #print(index)
            if pos_val == 0:
                ## What if test[y]==0 for pos_val=0????????????????
                if test_y[index]>0:
                     #trades.values[index,:] = max_holding
                     buyAndSell.append('BUY')
                     shares.append(max_holding)
                     dates.append(df_prices.index[index])
                     pos_val= pos_val+max_holding
                elif test_y[index]<=0:
                    #trades.values[index,:] = min_holding
                    buyAndSell.append('SELL')
                    shares.append(min_holding)
                    dates.append(df_prices.index[index])
                    pos_val = pos_val+min_holding
            if pos_val == max_holding:
                if test_y[index]<0:
                    #trades.values[index,:] = min_trade
                    buyAndSell.append('SELL')
                    shares.append(min_trade)
                    dates.append(df_prices.index[index])
                    pos_val = pos_val+min_trade
                elif test_y[index] == 0:
                    #what if we get rid of this condition??????
                    #trades.values[index,:] = min_holding
                    buyAndSell.append('SELL')
                    shares.append(min_holding)
                    dates.append(df_prices.index[index])
                    pos_val = pos_val+min_holding
            if pos_val == min_holding:
                if test_y[index]>0:
                    #trades.values[index,:] = max_trade
                    buyAndSell.append('BUY')
                    shares.append(max_trade)
                    dates.append(df_prices.index[index])
                    pos_val = pos_val+max_trade
                elif test_y[index] == 0:
                    #trades.values[index,:] = max_holding
                    buyAndSell.append('BUY')
                    shares.append(max_holding)
                    dates.append(df_prices.index[index])
                    pos_val= pos_val+max_holding

            #temp_holding_total = temp_holding_total+shares[index]
            temp_holding.append(pos_val)
            temp_dates.append(df_prices.index[index])
        
     
        #print(temp_holding)
        '''
        print(len(df_prices.index))
        print(len(test_y))
        print(len(test_y))
        for index in range(len(df_prices.index)):
            if test_y[index]>=0.5:
                shares.append(1000-pos_val)
                dates.append(df_prices.index[index])
                pos_val=pos_val+1000-pos_val

            elif test_y[index]<=0.5:
                shares.append(-1000-pos_val)
                dates.append(df_prices.index[index])
                pos_val=pos_val-1000-pos_val
            else:
                shares.append(0)
                dates.append(df_prices.index[index])
                pos_val=pos_val+0
        '''

        #print(len(shares))
        #print(shares)

        """
        if pos_val == max_holding or :
            trades.values[df_prices.shape[0]-1,:]=min_holding
        elif pos_val == min_holding:
            trades.values[df_prices.shape[0]-1,:]=max_holding
        """
        #print(dates)
        #print(shares)

        trades = pd.DataFrame(data=shares, index=dates, columns=['orders'])
        #print(trades)
        return trades  	   		   	 			  		 			     			  	  		 	  	 		 			  		  				  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("One does not simply think up a strategy")  	