

import csv
import json

input_file="review.json"
output_file="review_refined.csv"

inp=open(input_file,"r")
out=open(output_file,"w")

writer=csv.writer(out)

data=json.loads("["+inp.read().replace("}\n{", "},\n{") + "]")

writer.writerow(data[0].keys())

i=0
for row in data:   
    words=data[i]['text'].split()
    counter=len(words)
    if counter>10:       # Filter out reviews with 10 or less word counts
        writer.writerow(row.values())
    i+=1
    
out.close();