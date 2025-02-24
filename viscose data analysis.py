import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
from matplotlib import pyplot
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

tecumseh_lines = []
dataset_plant = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [1])
for line in dataset_plant.index:
    try:
        if 'Tecumseh' in dataset_plant['Plant (Target)'][line]:
            tecumseh_lines.append(line)
            
    except TypeError:
        continue




dataset_civ = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [25,26])

for i in tecumseh_lines:
    dataset_civ = dataset_civ.drop(labels=i,axis = 0)
    
dataset_civ['CiV (%).1'].replace('', np.nan, inplace=True)
dataset_civ.dropna(subset=['CiV (%).1'], inplace=True)
dataset_civ['SiV (%).1'].replace('', np.nan, inplace=True)
dataset_civ.dropna(subset=['SiV (%).1'], inplace=True)
dataset_civ = dataset_civ.drop(labels=3798,axis = 0)
data_civ = dataset_civ.to_numpy()



dataset_pres = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [36,37])
for i in tecumseh_lines:
    dataset_pres = dataset_pres.drop(labels=i,axis = 0)
dataset_pres['Press Box Pressure'].replace('', np.nan, inplace=True)
dataset_pres.dropna(subset=['Press Box Pressure'], inplace=True)
dataset_pres['Press Hemi'].replace('', np.nan, inplace=True)
dataset_pres.dropna(subset=['Press Hemi'], inplace=True)
data_pres = dataset_pres.to_numpy()

dataset_HR = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [0,28,29])
for i in tecumseh_lines:
    dataset_HR = dataset_HR.drop(labels=i,axis = 0)
dataset_HR['HR'].replace('', np.nan, inplace=True)
dataset_HR.dropna(subset=['HR'], inplace=True)
dataset_HR['BFV (S)'].replace('', np.nan, inplace=True)
dataset_HR.dropna(subset=['BFV (S)'], inplace=True)
data_HR = dataset_HR.to_numpy()


dataset_BFV = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [29,30])
for i in tecumseh_lines:
    dataset_BFV = dataset_BFV.drop(labels=i,axis = 0)
dataset_BFV = dataset_BFV.drop(labels=1581,axis = 0)
dataset_BFV = dataset_BFV.drop(labels=1734,axis = 0)
dataset_BFV = dataset_BFV.drop(labels=1725,axis = 0)
dataset_BFV['BFV (S)'].replace('', np.nan, inplace=True)
dataset_BFV.dropna(subset=['BFV (S)'], inplace=True)
dataset_BFV['Rv'].replace('', np.nan, inplace=True)
dataset_BFV.dropna(subset=['Rv'], inplace=True)
data_BFV = dataset_BFV.to_numpy()

dataset_CIAC = read_excel('Viscose Testing Log (Alex).xlsx', skiprows=1,usecols = [23,24])
for i in tecumseh_lines:
    dataset_CIAC = dataset_CIAC.drop(labels=i,axis = 0)
dataset_CIAC['CiAC (%)'].replace('', np.nan, inplace=True)
dataset_CIAC.dropna(subset=['CiAC (%)'], inplace=True)
dataset_CIAC['SiAC (%)'].replace('', np.nan, inplace=True)
dataset_CIAC.dropna(subset=['SiAC (%)'], inplace=True)
data_CIAC = dataset_CIAC.to_numpy()
              
for line in dataset_civ.index:
    try:
        if '-' in dataset_civ['SiV (%).1'][line]:
            print(line)
            
    except TypeError:
        continue


def cluster(X):
    model = KMeans(n_clusters =2 )
    model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)

    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix,0],X[row_ix,1],s=7)

    plt.xlabel('CiAC (%)')
    plt.ylabel("SiAC (%)")
    pyplot.show()
    
scores = []
std_errors = []
def machine_learning(X,Y,model):
    seeds = np.arange(43)
    for seed in seeds:
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.30,random_state=seed)
        #random_state seed doesnt change r squared :)
        model = model
        model.fit(X_train,Y_train)
        predictions = model.predict(X_validation)
        scores.append(r2_score(Y_validation,predictions))
        
        error = np.array(scores).std()
        std_errors.append(error)
    error_avg = np.mean(std_errors)
    scores_avg = np.mean(scores)
    print(scores_avg)
    print(error_avg)

    return scores_avg

blender_lines = []
for line in dataset_HR.index:
    try:
        if 'Churn' not in dataset_HR['Origin'][line]:
            blender_lines.append(line)
            
    except TypeError:
        continue

print(len(blender_lines))
