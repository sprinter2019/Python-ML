
import pandas as pd


join_file = pd.read_csv("user.csv")
input_file= pd.read_csv("joined_Business_Review_refined.csv")

merged=input_file[['user_id','business_id','text','date']].merge(join_file
                 [['user_id','review_count','useful']], how='left', on='user_id')

merged.to_csv("joined_Business_Review_User_refined.csv", index=False)
