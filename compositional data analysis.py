from pandas import read_excel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from numpy import unique
from numpy import where
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
    

# Load data from Excel files
df_chem = read_excel('CompositionalData.xlsx',skiprows = 4,usecols = [0,1,2,3,4])
df_mech = read_excel('CompositionalData.xlsx',skiprows = 4,usecols = [0,10,11,12])




new_cellulose = []

for line in df_chem.index:
    try:
     if '-' in df_chem['Cellulose (wt %)'][line]:
             new = df_chem['Cellulose (wt %)'][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_cellulose.append(mean)
             
     else:
        wow = str(df_chem['Cellulose (wt %)'][line])
        old = wow.split('–')
        no = float(old[0])
        no2 = float(old[1])
        mean1 = (no + no2)/2
        new_cellulose.append(mean1)
        #works
       
         
         
        
         
             
         
         
    except TypeError:
        new_cellulose.append(float(df_chem['Cellulose (wt %)'][line]))




new_hemicellulose = []

for line in df_chem.index:
    try:
     if '-' in df_chem['Hemicelluloses (wt %)'][line]:
             new = df_chem['Hemicelluloses (wt %)'][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_hemicellulose.append(mean)
             
     
             
     else:
        wow = str(df_chem['Hemicelluloses (wt %)'][line])
        old = wow.split('–')
        no = float(old[0])
        no2 = float(old[1])
        mean1 = (no + no2)/2
        new_hemicellulose.append(mean1)
        #works    
         
    except TypeError:
        new_hemicellulose.append(float(df_chem['Hemicelluloses (wt %)'][line]))
        



new_lignin = []

for line in df_chem.index:
    try:
     if '-' in df_chem['Lignin (wt %)'][line]:
             new = df_chem['Lignin (wt %)'][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_lignin.append(mean)
             
         
         
    except TypeError:
        new_lignin.append(float(df_chem['Lignin (wt %)'][line]))



new_pectin = []

for line in df_chem.index:
    try:
     if '-' in df_chem['Pectins (wt %)'][line]:
             new = df_chem['Pectins (wt %)'][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_pectin.append(mean)
             
         
         
    except TypeError:
        new_pectin.append(float(df_chem['Pectins (wt %)'][line]))

new_tens = []

for line in df_mech.index:
    try:
     if '-' in df_mech['Tensile Strength (Mpa)'][line]:
             new = df_mech['Tensile Strength (Mpa)'][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_tens.append(mean)
     else:
         continue
             
         
         
    except TypeError:
        new_tens.append(float(df_mech['Tensile Strength (Mpa)'][line]))
        
        
new_youngs = []

for line in df_mech.index:
    try:
     if '-' in df_mech["Young's Modulus (Gpa)"][line]:
             new = df_mech["Young's Modulus (Gpa)"][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_youngs.append(mean)
    
             
         
         
    except TypeError:
        new_youngs.append(float(df_mech["Young's Modulus (Gpa)"][line]))
        
new_break= []

for line in df_mech.index:
    try:
     if '-' in df_mech["Elongation at Break (%)"][line]:
             new = df_mech["Elongation at Break (%)"][line].split('-')
             yes = float(new[0])
             yes2 = float(new[1])
             mean = (yes + yes2)/2
             new_break.append(mean)
             
         
         
    except TypeError:
        new_break.append(float(df_mech["Elongation at Break (%)"][line]))
    except ValueError:
        new_break.append(0)

def clustering(X,Y):
    """
    Performs KMeans clustering on the provided dataset and visualizes the clusters.

    Args:
        X (array): Independent variable(s) data.
        Y (array): Dependent variable(s) data.
    """
    values = np.concatenate([X[0:20].reshape(-1,1),Y[0:20].reshape(-1,1)],axis=1)
    model = KMeans(n_clusters=2)
    # fit the model
    model.fit(values)
    # assign a cluster to each example
    yhat = model.predict(values)
    print(yhat)
    # retrieve unique clusters
    clusters = unique(yhat)
    print(clusters)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
    	# get row indexes for samples with this cluster
    	row_ix = where(yhat == cluster)
    	# create scatter of these samples
    	plt.scatter(values[row_ix, 0], values[row_ix, 1])
    # show the plot
    plt.show()
    print(X[row_ix, 1])

scores = []
std_errors = []
def machine_learning(X,Y,model):
    """
    Performs machine learning regression analysis using cross-validation.

    Args:
        X (array): Independent variable(s) data.
        Y (array): Dependent variable(s) data.
        model (sklearn model): The regression model (e.g., LinearRegression).
    """
    seeds = np.arange(43)
    for seed in seeds:
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.40,random_state=seed)
        #random_state seed doesnt change r squared :)
        model = model
        model.fit(X_train,Y_train)
        predictions = model.predict(X_validation)
        scores.append(r2_score(Y_validation,predictions))
        
        error = np.array(scores).std()
        std_errors.append(error)
    error_avg = np.mean(std_errors)
    scores_avg = np.mean(scores)
    print(scores_avg,"±",error_avg)

# Convert lists to NumPy arrays
cell_arr = np.array(new_cellulose)
hemicell_arr = np.array(new_hemicellulose)

cell_hemi = np.concatenate((cell_arr.reshape(-1,1),hemicell_arr.reshape(-1,1)),axis=1)
cell_hemi = np.delete(cell_hemi,19,0)

lig_arr = np.array(new_lignin)
pec_arr = np.array(new_pectin)

tens_arr = np.array(new_tens)
tens_arr = np.delete(tens_arr,9,0)
youngs_arr = np.array(new_youngs)
youngs_arr = np.delete(youngs_arr,8,0)

#note supervised machine learning doesnt work between these two.




def cluster(X):
    """
    Clusters the given data using KMeans and visualizes it.
    """
    model = KMeans(n_clusters = 2)
    model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)

    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix,0],X[row_ix,1],s=15)
    plt.xlabel("Cellulose (wt%)")
    plt.ylabel("Tensile Strength (Mpa)")
    plt.title("",fontweight = 'bold')
    pyplot.show()
    
def scatter(dat):
    """
    Creates a scatter plot to visualize different material categories.
    """
    fig, ax = plt.subplots()
    ax.plot(dat[0:11,0],dat[0:11,1], 'x',color = 'salmon',label = 'Bast')
    ax.plot(dat[11:19,0],dat[11:19,1], 'x',color = 'palegreen',label = 'Leaf')
    ax.plot(dat[19:24,0],dat[19:24,1], 'x',color = 'palegoldenrod',label ='Seed' )
    ax.plot(dat[24:26,0],dat[24:26,1], 'x',color = 'lightskyblue',label = 'Straw')
    ax.plot(dat[26:30,0],dat[26:30,1], 'x',color = 'red',label = 'Grass')
    ax.plot(dat[30:33,0],dat[30:33,1], 'x',color = 'darkviolet',label = 'Softwoods')
    ax.plot(dat[33:37,0],dat[33:37,1], 'x',color = 'navy',label = 'Hardwoods')
    plt.xlabel('Celluose (wt%)')
    plt.ylabel('Hemicelluoses (wt%)')
    ax.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.show()

def impact_plot(X,Y):
    """
    Creates a linear regression plot showing the relationship between X and Y.
    """
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X,Y, 'x')
    plt.plot(X, m*X + b)
    plt.title('')
    plt.xlabel('Cellulose (wt%)')
    plt.ylabel("Tensile Strength (Mpa)")
    plt.annotate('y = {m}x + {c}'.format(m=round(m,3),c=round(b,3)),(40,1000),color = "red")
    plot = plt.show()
    return plot

# Prepare data for clustering and analysis
cell_arr2 = np.delete(cell_arr,[8,24,26,31,32,33,34,35,36,37],0)
hemicell_arr2 = np.delete(hemicell_arr,[8,24,26,31,32,33,34,35,36,37],0)
cell_ym = np.concatenate((cell_arr2.reshape(-1,1),youngs_arr.reshape(-1,1)),axis=1)
hemicell_tens = np.concatenate((hemicell_arr2.reshape(-1,1),tens_arr.reshape(-1,1)),axis=1)


cell_arr3 = np.delete(cell_arr,[9,24,26,31,32,33,34,35,36,37],0)
hemicell_arr3 = np.delete(hemicell_arr,[9,24,26,31,32,33,34,35,36,37],0)
cell_tens = np.concatenate((cell_arr3.reshape(-1,1),tens_arr.reshape(-1,1)),axis=1)
hemicell_tens = np.concatenate((hemicell_arr3.reshape(-1,1),tens_arr.reshape(-1,1)),axis=1)





# Execute clustering and machine learning models
cluster(cell_tens)
impact_plot(cell_arr3,tens_arr)
machine_learning(cell_arr3.reshape(-1,1), tens_arr.reshape(-1,1), LinearRegression())
