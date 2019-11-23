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

###########################
def binning_seats(data):
    seats_data = []
    for i in range(len(data)):
        seats_data.append(data["seats"][i])
    return max(seats_data) - min(seats_data) + 1
	

def make_full_lists(data):
    all_seats = []
    all_mms = []
    all_pgs = []
    for i in range(len(data)):
        all_seats.append(data["seats"][i])
        all_mms.append(data["mm"][i])
        all_pgs.append(data["pg"][i])
    return all_seats, all_pgs, all_mms	


def make_winnowed_lists(data, inliers, pg_bound, mm_bound):
    mm_routliers = []
    mm_doutliers = []
    pg_routliers = []
    pg_doutliers = []
    vs_routliers = []
    vs_doutliers = []
    ideal_pg_seats = []
    ideal_mm_seats = []

    for i in range(len(data)):
        if data["seats"][i] < inliers[0]:
            mm_doutliers.append(data["mm"][i])
            pg_doutliers.append(data["pg"][i])
            vs_doutliers.append(data["vs"][i])
        if data["seats"][i] > inliers[1]:
            mm_routliers.append(data["mm"][i])
            pg_routliers.append(data["pg"][i])
            vs_routliers.append(data["vs"][i])
        if data["pg"][i] < pg_bound:
            ideal_pg_seats.append(data["seats"][i])
        if abs(data["mm"][i]) < mm_bound:
            ideal_mm_seats.append(data["seats"][i])

    outlying_mms = [mm_routliers, mm_doutliers]
    outlying_pgs = [pg_routliers, pg_doutliers]

    return outlying_pgs, outlying_mms, len(mm_doutliers) + len(mm_routliers), ideal_pg_seats, ideal_mm_seats
	
########################################################
def visualize(state, data, inliers, num_dist, pg_bound, mm_bound,state_name,election_name,bound_name):
    
    # Setting up figures
    fig1, seats = plt.subplots(figsize=(16,4))
    fig2, pg_seats = plt.subplots(figsize=(16,4))
    fig3, mm_seats = plt.subplots(figsize=(16,4))
    fig4, ax = plt.subplots(2,2, figsize=(16,8), sharex="col", sharey="row")
    
    seats_figs = [seats, pg_seats, mm_seats]
    
    # Spacing of subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Colors
    party_colors = ["red", "blue"]
    neutral = "lightgreen"
    
    # Variables
    bins_num = 20
    out_bins_num = binning_seats(data)
    num_steps = len(data)
    
    # Formatting data
    all_seats, all_pgs, all_mms = make_full_lists(data)
    outlying_pgs, outlying_mms, num_outliers, ideal_pg_seats, ideal_mm_seats = make_winnowed_lists(data, inliers, pg_bound, mm_bound)
    
    # More Variables
    num_best_pgs = len(ideal_pg_seats)
    num_best_mms = len(ideal_mm_seats)
    pct_outliers = round(num_outliers / num_steps * 100, 1)
    pct_best_pgs = round((num_best_pgs/num_steps) * 100,1)
    pct_best_mms = round((num_best_mms/num_steps) * 100,1)
    
    custom_lines = [[Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color=neutral, lw=4),
                    Line2D([0], [0], color="red", lw=4)],
                   [Line2D([0], [0], color="lightblue", lw=4),
                    Line2D([0], [0], color=neutral, lw=4),
                    Line2D([0], [0], color="lightcoral", lw=4)],
                   [Line2D([0], [0], color="lightblue", lw=4),
                    Line2D([0], [0], color=neutral, lw=4),
                    Line2D([0], [0], color="lightcoral", lw=4)]]
    
    outlier_labels = ["Dem. leaning plans", "Neutral plans", "Rep. leaning plans"]
    
    # Titles 
    all_plans_title = state + ": " + str(num_steps) + " ReCom plans"
    outlying_seats_title = state + ": " + str(num_outliers) + " outlier plans " + "(" + str(pct_outliers) + "%)"
    all_seats_title = state + ": Republican seats won: " + str(num_steps) + " ReCom plans" 
    ideal_pgs_title = state + " Republican seats won: " + str(num_best_pgs) + " plans with PG < " + str(pg_bound) + " (" + str(pct_best_pgs) + "% of all plans)"
    ideal_mms_title = state + " Republican seats won: " + str(num_best_mms) + " plans with |MM| < " + str(mm_bound) + " (" + str(pct_best_mms) + "% of all plans)"
    
    seats_titles = [all_seats_title, ideal_pgs_title, ideal_mms_title]
    
    # Plotting the data into the figures
    n, bins, patches = seats.hist(all_seats, bins=range(num_dist+2), color=neutral)#was plus 1 missed for UT?
    for i in range(num_dist):
        if i < inliers[0]:
            patches[i].set_fc("blue")
        if i > inliers[1]:
            patches[i].set_fc("red")
    n, bins, patches = pg_seats.hist(ideal_pg_seats, bins=range(num_dist+2), color=neutral)#was plus 1 missed for UT?
    for i in range(num_dist):
        if i < inliers[0]:
            patches[i].set_fc("lightblue")
        if i > inliers[1]:
            patches[i].set_fc("lightcoral")
    n, bins, patches = mm_seats.hist(ideal_mm_seats, bins=range(num_dist+2), color=neutral)#was plus 1 missed for UT?
    for i in range(num_dist):
        if i < inliers[0]:
            patches[i].set_fc("lightblue")
        if i > inliers[1]:
            patches[i].set_fc("lightcoral")
            
    ax[0,0].hist(all_mms, bins=2*bins_num, color=neutral)
    ax[0,1].hist(all_pgs, bins=2*bins_num, color=neutral)
    ax[1,0].hist(outlying_mms, bins=bins_num, color=party_colors)
    ax[1,1].hist(outlying_pgs, bins=bins_num, color=party_colors)

    
    # Aesthetics for figures
    for i in seats_figs:
        i.grid()
        i.set_ylabel("Frequency")
        i.set_xlabel("# Seats Won")
        i.set_xlim(0, num_dist+1)
        i.set_xticks(range(num_dist + 2))
        i.set_xticklabels(range(num_dist + 1))
        i.axvline(x=(num_dist+1)/2, color="black", linestyle="dashed", linewidth=2, label="50% seats")
        # NEED TO PUT IN PROPORTIONAL MARKER
        
    for i in range(3):
        seats_figs[i].set_title(seats_titles[i])
        leg1 = seats_figs[i].legend(custom_lines[i], outlier_labels)
        leg2 = seats_figs[i].legend(loc="upper left")
        seats_figs[i].add_artist(leg1)
        
#     # Add first legend:  only labeled data is included
# leg1 = ax.legend(loc='lower left')
# # Add second legend for the maxes and mins.
# # leg1 will be removed from figure
# leg2 = ax.legend([maxline,minline],['max','min'], loc='upper right')
# # Manually add the first legend back
# ax.add_artist(leg1)
        
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].grid()
            ax[i,j].set_ylabel("Frequency")
            ax[i,j].axvline(x=0, color="black", linewidth=2, linestyle="dashed", label="ideal value")
            ax[i,j].tick_params(labelleft=True, labelbottom=True)
            ax[i,j].legend()
            if j == 0:
                ax[i,j].set_xlabel("Mean-Median Score")
            else:
                ax[i,j].set_xlabel("Partisan Gini Score")
            if i == 0:
                ax[i,j].set_title(all_plans_title)
            else:
                ax[i,j].set_title(outlying_seats_title)	

    fig1.savefig(newdir+state_name+election_name+"_"+bound_name+"_seats.png")
    fig2.savefig(newdir+state_name+election_name+"_"+bound_name+"_pgseats.png")
    fig3.savefig(newdir+state_name+election_name+"_"+bound_name+"_mmseats.png")
    fig4.savefig(newdir+state_name+election_name+"_"+bound_name+"_all.png")
	
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

########################################################

"""
election_name = "SEN14"
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
	
############################################################
visualize(state_abbr+"-"+election_name, df, [7,8], 13, 0.01, 0.001,state_name,election_name)
"""
###########################################################################################

elects = [ ["GOV08", "SEN08", "SEN10", "GOV12", "SEN14", "PRES12", "PRES16", "SEN16", "GOV16"],
 ["PRES12", "SEN12", "SEN14", "GOV14", "PRES16"],
 ['PRES16', 'SEN16', 'GOV16']]
 
abbrs = ["NC", "TX", "UT"]

names=["northcarolina","texas","utah"]

#bounds = [[7,8],[22, 27],[4,4]]

bounds = [
[
[[4,5],[4,5]],
[[2,5],[2,4]],
[[9,11],[10,11]],
[[8,11],[9,10]],
[[6,9],[7,8]],
[[6,9],[7,8]],
[[6,9],[7,9]],
[[7,10],[8,9]],
[[6,8],[7,8]]
],
[
[[21,28],[22,27]],
[[22,27],[23,26]],
[[25,30],[26,29]],
[[22,28],[23,28]],
[[18,23],[19,22]]
],
[
[[3,3],[3,3]],
[[4,4],[4,4]],
[[4,4],[4,4]]
]
]

districts = [13,36,4]

steps = [100000,100000,50000]

pgvalue = [.01,.03,.01]

bound_labels=["strong","weak"]

for i in range(3):
	temp = 0
	for ename in elects[i]:
		for bound_name in [0,1]:
		    
	
	
			print("Plotting",names[i],ename,"type",bound_labels[bound_name])
			print("Bounds",bounds[i][temp][bound_name])
		
			election_name = ename
			state_name = names[i]
			state_abbr = abbrs[i]

			datadir = "./" + state_abbr + "output/"



			newdir = "./Paper_plots2/"
			os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
			with open(newdir + "init.txt", "w") as f:
				f.write("Created Folder")
			

			max_steps = steps[i]#100000
			step_size = 10000#2000#2000

			ts = [x*step_size for x in range(1,int(max_steps/step_size)+1)]
				
				

			df = pd.DataFrame(columns = ['seats','mm','pg','vs','eg','ce'])


			for t in ts:
				tempdf=pd.read_csv(datadir + state_name + election_name +"_data"+str(t)+".csv", delimiter=',')
				
				df = pd.concat([df, tempdf], ignore_index=True)

			
			
				visualize(abbrs[i]+"-"+ename, df, bounds[i][temp][bound_name], districts[i], pgvalue[i], 0.001, names[i], ename,bound_labels[bound_name])
	
		temp+=1



