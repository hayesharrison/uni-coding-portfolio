
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from numpy import unique
from numpy import where

max_ext_1_raw = read_excel('project data.xlsx',skiprows=17,skipfooter= 35,usecols = [1,5,9,13,17,21,25,29])
max_ext_1 = max_ext_1_raw.values
max_ext_100_raw = read_excel('project data.xlsx',skiprows=24,skipfooter= 28,usecols = [1,5,9,13,17,21,25,29])
max_ext_100 = max_ext_100_raw.values
max_ext_300_raw = read_excel('project data.xlsx',skiprows=32,skipfooter= 20,usecols = [1,5,9,13,17,21,25,29])
max_ext_300 = max_ext_300_raw.values
max_ext_1000_raw = read_excel('project data.xlsx',skiprows=40,skipfooter= 12,usecols = [1,5,9,13,17,21,25,29])
max_ext_1000 = max_ext_1000_raw.values
max_ext_120000_raw = read_excel('project data.xlsx',skiprows=48,skipfooter= 4,usecols = [1,5,9,13,17,21,25,29])
max_ext_120000 = max_ext_120000_raw.values
max_ext_all = np.concatenate((max_ext_1.reshape(-1,1),max_ext_100.reshape(-1,1),max_ext_300.reshape(-1,1),max_ext_1000.reshape(-1,1)))


max_force_1_raw = read_excel('project data.xlsx',skiprows=18,skipfooter= 34,usecols = [1,5,9,13,17,21,25,29])
max_force_1 = max_force_1_raw.values
max_force_100_raw = read_excel('project data.xlsx',skiprows=25,skipfooter= 27,usecols = [1,5,9,13,17,21,25,29])
max_force_100 = max_force_100_raw.values
max_force_300_raw = read_excel('project data.xlsx',skiprows=33,skipfooter= 19,usecols = [1,5,9,13,17,21,25,29])
max_force_300 = max_force_300_raw.values
max_force_1000_raw = read_excel('project data.xlsx',skiprows=41,skipfooter= 11,usecols = [1,5,9,13,17,21,25,29])
max_force_1000 = max_force_1000_raw.values
max_force_120000_raw = read_excel('project data.xlsx',skiprows=49,skipfooter= 3,usecols = [1,5,9,13,17,21,25,29])
max_force_120000 = max_force_120000_raw.values
max_force_all = np.concatenate((max_force_1.reshape(-1,1),max_force_100.reshape(-1,1),max_force_300.reshape(-1,1),max_force_1000.reshape(-1,1)))

norm_energy_1_raw = read_excel('project data.xlsx',skiprows=19,skipfooter= 33,usecols = [1,5,9,13,17,21,25,29])
norm_energy_1 = norm_energy_1_raw.values
norm_energy_100_raw = read_excel('project data.xlsx',skiprows=26,skipfooter= 26,usecols = [1,5,9,13,17,21,25,29])
norm_energy_100 = norm_energy_100_raw.values
norm_energy_300_raw = read_excel('project data.xlsx',skiprows=34,skipfooter= 18,usecols = [1,5,9,13,17,21,25,29])
norm_energy_300 = norm_energy_300_raw.values
norm_energy_1000_raw = read_excel('project data.xlsx',skiprows=42,skipfooter= 10,usecols = [1,5,9,13,17,21,25,29])
norm_energy_1000 = norm_energy_1000_raw.values
norm_energy_120000_raw = read_excel('project data.xlsx',skiprows=50,skipfooter= 2,usecols = [1,5,9,13,17,21,25,29])
norm_energy_120000 = norm_energy_120000_raw.values
norm_energy_all = np.concatenate((norm_energy_1,norm_energy_100,norm_energy_300,norm_energy_1000,norm_energy_120000))

mod_1_raw = read_excel('project data.xlsx',skiprows=20,skipfooter= 32,usecols = [1,5,9,13,17,21,25,29])
mod_1 = mod_1_raw.values
mod_100_raw = read_excel('project data.xlsx',skiprows=27,skipfooter= 25,usecols = [1,5,9,13,17,21,25,29])
mod_100 = mod_100_raw.values
mod_300_raw = read_excel('project data.xlsx',skiprows=35,skipfooter= 17,usecols = [1,5,9,13,17,21,25,29])
mod_300 = mod_300_raw.values
mod_1000_raw = read_excel('project data.xlsx',skiprows=43,skipfooter= 9,usecols = [1,5,9,13,17,21,25,29])
mod_1000 = mod_1000_raw.values
mod_120000_raw= read_excel('project data.xlsx',skiprows=51,skipfooter= 1,usecols = [1,5,9,13,17,21,25,29])
mod_120000 = mod_120000_raw.values
mod_all = np.concatenate((mod_1.reshape(-1,1),mod_100.reshape(-1,1),mod_300.reshape(-1,1),mod_1000.reshape(-1,1)))


ext_index_list = [1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35]
force_index_list = [0,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35]
mod_index_list = [0,1,2,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35]  


sample_one_ext = read_excel('project data.xlsx',skiprows=17,usecols = [1])
for line in ext_index_list:
    sample_one_ext = sample_one_ext.drop(labels=line,axis = 0)
sample_two_ext = read_excel('project data.xlsx',skiprows=17,usecols = [5])
for line in ext_index_list:
    sample_two_ext = sample_two_ext.drop(labels=line,axis = 0)
sample_thr_ext = read_excel('project data.xlsx',skiprows=17,usecols = [9])
for line in ext_index_list:
    sample_thr_ext = sample_thr_ext.drop(labels=line,axis = 0)   
sample_four_ext = read_excel('project data.xlsx',skiprows=17,usecols = [13])
for line in ext_index_list:
    sample_four_ext = sample_four_ext.drop(labels=line,axis = 0)  
sample_five_ext = read_excel('project data.xlsx',skiprows=17,usecols = [17])
for line in ext_index_list:
    sample_five_ext = sample_five_ext.drop(labels=line,axis = 0)  
sample_six_ext = read_excel('project data.xlsx',skiprows=17,usecols = [21])
for line in ext_index_list:
    sample_six_ext = sample_six_ext.drop(labels=line,axis = 0)  
sample_agri_ext = read_excel('project data.xlsx',skiprows=17,usecols = [25])
for line in ext_index_list:
    sample_agri_ext = sample_agri_ext.drop(labels=line,axis = 0)  
sample_pf_ext = read_excel('project data.xlsx',skiprows=17,usecols = [29])
for line in ext_index_list:
    sample_pf_ext = sample_pf_ext.drop(labels=line,axis = 0) 

sample_one_force = read_excel('project data.xlsx',skiprows=17,usecols = [1])
for line in force_index_list:
    sample_one_force = sample_one_force.drop(labels=line,axis = 0)
sample_two_force = read_excel('project data.xlsx',skiprows=17,usecols = [5])
for line in force_index_list:
    sample_two_force = sample_two_force.drop(labels=line,axis = 0)
sample_thr_force = read_excel('project data.xlsx',skiprows=17,usecols = [9])
for line in force_index_list:
    sample_thr_force = sample_thr_force.drop(labels=line,axis = 0)   
sample_four_force =  read_excel('project data.xlsx',skiprows=17,usecols = [13])
for line in force_index_list:
    sample_four_force = sample_four_force.drop(labels=line,axis = 0)  
sample_five_force = read_excel('project data.xlsx',skiprows=17,usecols = [17])
for line in force_index_list:
    sample_five_force = sample_five_force.drop(labels=line,axis = 0)  
sample_six_force = read_excel('project data.xlsx',skiprows=17,usecols = [21])
for line in force_index_list:
    sample_six_force = sample_six_force.drop(labels=line,axis = 0)  
sample_agri_force = read_excel('project data.xlsx',skiprows=17,usecols = [25])
for line in force_index_list:
    sample_agri_force = sample_agri_force.drop(labels=line,axis = 0)  
sample_pf_force= read_excel('project data.xlsx',skiprows=17,usecols = [29])
for line in force_index_list:
    sample_pf_force = sample_pf_force.drop(labels=line,axis = 0) 


sample_one_mod= read_excel('project data.xlsx',skiprows=17,usecols = [1])
for line in mod_index_list:
    sample_one_mod = sample_one_mod.drop(labels=line,axis = 0)
sample_two_mod = read_excel('project data.xlsx',skiprows=17,usecols = [5])
for line in mod_index_list:
    sample_two_mod = sample_two_mod.drop(labels=line,axis = 0)
sample_thr_mod = read_excel('project data.xlsx',skiprows=17,usecols = [9])
for line in mod_index_list:
    sample_thr_mod = sample_thr_mod.drop(labels=line,axis = 0)   
sample_four_mod =  read_excel('project data.xlsx',skiprows=17,usecols = [13])
for line in mod_index_list:
    sample_four_mod = sample_four_mod.drop(labels=line,axis = 0)  
sample_five_mod = read_excel('project data.xlsx',skiprows=17,usecols = [17])
for line in mod_index_list:
    sample_five_mod = sample_five_mod.drop(labels=line,axis = 0)  
sample_six_mod = read_excel('project data.xlsx',skiprows=17,usecols = [21])
for line in mod_index_list:
    sample_six_mod = sample_six_mod.drop(labels=line,axis = 0)  
sample_agri_mod = read_excel('project data.xlsx',skiprows=17,usecols = [25])
for line in mod_index_list:
    sample_agri_mod = sample_agri_mod.drop(labels=line,axis = 0)  
sample_pf_mod = read_excel('project data.xlsx',skiprows=17,usecols = [29])
for line in mod_index_list:
    sample_pf_mod = sample_pf_mod.drop(labels=line,axis = 0) 






def puncture_plt_samples():
    fig, ax = plt.subplots()
    ax.plot(sample_one_mod,sample_one_force, 'x',color = 'red',label = 'Sample 1')
    ax.plot(sample_two_mod,sample_two_force, 'x',color = 'deepskyblue',label ='Sample 2' )
    ax.plot(sample_thr_mod,sample_thr_force, 'x',color = 'green',label ='Sample 3' )
    ax.plot(sample_four_mod,sample_four_force, 'x',color = 'orange',label = 'Sample 4')
    ax.plot(sample_five_mod,sample_five_force, 'x',color = 'yellow',label = 'Sample 5')
    ax.plot(sample_six_mod,sample_six_force, 'x',color = 'purple',label = 'Sample 6')
    ax.plot(sample_agri_mod,sample_agri_force, 'x',color = 'black',label = 'AGRI')
    ax.plot(sample_pf_mod,sample_pf_force, 'x',color = 'navy',label = 'PF')
    plt.xlabel('Modulus N/mm')
    plt.ylabel('Max Force N/mm')
    ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.show()

def puncture_plt(X,Y):
    fig, ax = plt.subplots()
    ax.plot(X[0,0:8],Y[0,0:8], 'x',color = 'red',label = '1mm/min')
    ax.plot(X[1,0:8],Y[1,0:8], 'x',color = 'blue',label ='100mm/min' )
    ax.plot(X[2,0:8],Y[2,0:8], 'x',color = 'green',label ='300mm/min' )
    ax.plot(X[3,0:8],Y[3,0:8], 'x',color = 'orange',label = '1000mm/min')
    plt.xlabel('Modulus')
    plt.ylabel('Maximum Force N/mm')
    ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.show()
    
    
    
def cluster(X):
    model = KMeans(n_clusters=4)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
    	# get row indexes for samples with this cluster
    	row_ix = where(yhat == cluster)
    	# create scatter of these samples
    	plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.xlabel('Max Extension (mm)')
    plt.ylabel("Modulus (N/mm)")
    plt.show()

ext_mod = np.concatenate((max_ext_all,mod_all),axis = 1)

mod_force =  np.concatenate((mod_all,max_force_all),axis = 1)

print(sample_pf_mod)



cluster(ext_mod)

puncture_plt(ext_mod)




