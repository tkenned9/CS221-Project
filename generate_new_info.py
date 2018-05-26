import os
import pandas as pd
cwd = os.getcwd

file = "all_data_info.csv"
print "reading data"
data_info = pd.read_csv(file)
names = ["Sofonisba Anguissola", "Leonardo da Vinci","Titian", "Parmigianino","Tintoretto"]
print "selecting paintings"
new_data_info = data_info[data_info["artist"].isin(names)]
"exporting to csv"
new_data_info.to_csv("new_data_info.csv", encoding='utf-8')





