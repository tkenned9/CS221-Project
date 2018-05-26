import os
import pandas as pd
cwd = os.getcwd()
cwd += "/"

file = "new_data_info.csv"
data_info = pd.read_csv(file)
names = ["Sofonisba Anguissola", "Leonardo da Vinci","Titian", "Parmigianino","Tintoretto"]
data_info = data_info.sample(frac=1)

count = 0
num_rows = data_info.shape[0]
print num_rows
train_frac = .8
dev_frac = .9
for row in data_info.itertuples():
	targ_directory = cwd
	if count < train_frac*num_rows:
		targ_directory += "training/"
		print "t"
	elif count < dev_frac*num_rows:
		targ_directory += "validation/"
		print "v"
	else:
		targ_directory += "test/"

	name = row.artist


	
	if name == "Leonardo da Vinci": 
		name = "Leonardo_da_Vinci"
	if name == "Sofonisba Anguissola":
		name = "Sofonisba_Anguissola"
	targ_directory += name
	targ_directory += "/"


	source_directory = cwd
	if row.in_train == True:
		source_directory += "old_train/"
	elif row.in_train == False:
		source_directory += "old_test/"


	source_directory += row.new_filename
	os.system('cp %s %s' %(source_directory,targ_directory))
	count += 1
	print count

