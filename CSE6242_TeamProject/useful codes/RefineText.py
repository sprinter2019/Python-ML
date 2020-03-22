
import pandas as pd
import numpy as np
from textblob import TextBlob

input_dataframe= pd.read_csv("joined_Business_Review_User_refined.csv")


def sensitivity_analysis(text):
    return (TextBlob(str(text))).sentiment.polarity
    
TextScore=pd.DataFrame(columns=['business_id','polarity'])

for i in range(len(input_dataframe)):
    print(i)
    
    col1=str(input_dataframe['business_id'][i])
    col2=sensitivity_analysis(input_dataframe['text'][i])
    
    df=pd.DataFrame([[col1, col2]], columns=['business_id','polarity'])
    
    TextScore=TextScore.append(df,ignore_index=True)

meanTextScore=TextScore.groupby('business_id').aggregate(np.mean)

TextScore.to_csv("TextPolarity.csv", index=False)
meanTextScore.to_csv("AvgTextPolarity.csv", index=False)
