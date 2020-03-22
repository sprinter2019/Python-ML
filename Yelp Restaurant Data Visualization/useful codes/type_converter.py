

import csv
import json

input_file="user.json"
output_file="user.csv"

inp=open(input_file,"r")
out=open(output_file,"w")

writer=csv.writer(out)

data=json.loads("["+inp.read().replace("}\n{", "},\n{") + "]");

writer.writerow(data[0].keys())

for row in data:   
    writer.writerow(row.values())
    
out.close();