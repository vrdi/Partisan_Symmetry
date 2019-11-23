import geopandas as gpd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import pandas as pd

import os
from functools import partial
import json
import csv

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import random
import seaborn as sns



election_name = "SEN16"
state_name = "northcarolina"
state_abbr = "NC"

datadir = "./" + state_abbr + "output/"



newdir = "./Paper_plots/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
	f.write("Created Folder")
	

max_steps = 100000
step_size = 10000#2000#2000

ts = [x*step_size for x in range(1,int(max_steps/step_size)+1)]
	
	

df = pd.DataFrame(columns = ['seats','mm','pg','vs','eg','ce'])


for t in ts:
	tempdf=pd.read_csv(datadir + state_name + election_name +"_data"+str(t)+".csv", delimiter=',')
	
	df = pd.concat([df, tempdf], ignore_index=True)


election_name = "PRES16"
state_name = "northcarolina"
state_abbr = "NC"

datadir = "./" + state_abbr + "output/"



newdir = "./Paper_plots/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
	f.write("Created Folder")
	

max_steps = 100000
step_size = 10000#2000#2000

ts = [x*step_size for x in range(1,int(max_steps/step_size)+1)]
	
	

df2 = pd.DataFrame(columns = ['seats','mm','pg','vs','eg','ce'])


for t in ts:
	tempdf=pd.read_csv(datadir + state_name + election_name +"_data"+str(t)+".csv", delimiter=',')
	
	df2 = pd.concat([df2, tempdf], ignore_index=True)
	
temp1 = list(df["mm"])
temp2 = list(df2["mm"])


max_difference = 0 

mismatches = 0
for i in range(len(temp1)):
	if temp1[i] < 0:
		if temp2[i] > 0:
			mismatches += 1
			if abs(temp1[i] - temp2[i]) > max_difference:
				max_difference = abs(temp1[i] - temp2[i])
			
			
	if temp1[i] > 0:
		if temp2[i] < 0:
			mismatches += 1
			if abs(temp1[i] - temp2[i]) > max_difference:
				max_difference = abs(temp1[i] - temp2[i])
			
#	if abs(temp1[i] - temp2[i]) > max_difference:#
#		max_difference = abs(temp1[i] - temp2[i])

print(mismatches)
print(max_difference)
