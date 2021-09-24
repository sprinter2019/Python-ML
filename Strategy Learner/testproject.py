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
import experiment1 as exp1
import experiment2 as exp2

def author():
    return "mhassan49"

if __name__ == "__main__":
    #random.seed(903450999)
    #np.random.seed(903450999)
    random.seed(1481090000)
    np.random.seed(1481090000)

    ############# Manual Strategy Assessment #########
    ms.test_code()

    ############ Experiment-1 #################
    exp1.test_code()

    ############ Experiment-2 #################
    exp2.test_code()