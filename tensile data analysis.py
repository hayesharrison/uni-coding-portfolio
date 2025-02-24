import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans


    



#loading in a combining MD and TD data
dataset = read_excel('project data.xlsx',skipfooter=42, skiprows=3,usecols = 
[1,3,5,7,9,11])
td_dataset = read_excel('project data.xlsx',skipfooter=42, skiprows=3,usecols = 
[1,3,5])
td_values = td_dataset.values
md_dataset = read_excel('project data.xlsx',skipfooter=42, skiprows=3,usecols = 
[7,9,11])
md_values = md_dataset.values
#reading in error values
error_dataset = read_excel('project data.xlsx',skipfooter=42, skiprows=3,usecols = 
[2,4,6,8,10,12])
errors = error_dataset.values
#assigning columns of data set to individual arrays
youngs_td = td_values[:,0]
youngs_td_error = errors[:,0]
youngs_md = md_values[:,0]
youngs_md_error = errors[:,3]
youngs_modulus =  np.concatenate((youngs_td,youngs_md))
youngs_error = np.concatenate((youngs_td_error,youngs_md_error))
strength_failure = np.concatenate((td_values[:,-2:],md_values[:,-2:]))

strength = strength_failure[:,0].reshape(-1,1)
strength_td_error = errors[:,1]
strength_md_error = errors[:,4]
strength_error = ym_error = np.concatenate((strength_td_error,strength_md_error))
failure  = strength_failure[:,1].reshape(-1,1)
failure_td_error = errors[:,2]
failure_md_error = errors[:,5]
failure_error = np.concatenate((failure_td_error,failure_md_error))
ln_youngs = np.log(youngs_modulus)
ln_strength = np.log(strength)
ln_failure = np.log(failure)
ln_strengthfailure = np.concatenate((ln_strength,ln_failure),axis=1)
strength_youngs = np.concatenate((strength,youngs_modulus.reshape(-1,1)),axis=1)
failure_youngs = np.concatenate((failure,youngs_modulus.reshape(-1,1)),axis=1)

#initial plots to look at data
def peek_plots(td,md):
    plot_dataset = pandas.DataFrame(np.concatenate((td,md)),columns = ['Youngs Modulus','Strength','Failure'])
    plot_dataset.plot(kind='box',subplots=True, layout=(3,3),sharex=False,sharey=False)
    pyplot.show()
    plot_dataset.hist()
    pyplot.show()
    scatter_matrix(plot_dataset)
    pyplot.show()

def strength_vs_youngs(X,Y,X_err,Y_err):
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, Y, 'x')
    plt.plot(X, m*X + b)
    plt.title('Strength vs Youngs Modulus (MD and TD data)')
    plt.xlabel('Strength (MPa)')
    plt.ylabel("Young's Modulus (GPa)")
    plt.errorbar(X,Y,Y_err,X_err,ls='none',ecolor = 'lightskyblue',capsize = 4)
    plt.annotate('y = {m}x + {c}'.format(m=round(m,3),c=round(b,3)),(60,6.5),color = "red")
    plot = plt.show()
    return plot
    
def failure_vs_youngs(X,Y,X_err,Y_err):
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, Y, 'x')
    plt.plot(X, m*X + b)
    plt.title('Failure vs Youngs Modulus (MD and TD data)')
    plt.xlabel('Elongation of sample at failure (%)')
    plt.ylabel("Young's Modulus (GPa)")
    plt.errorbar(X,Y,Y_err,X_err, ls = 'none',ecolor = 'lightskyblue',capsize=4)
    plt.annotate("Sample 1", (0.44,7.137273333))
    plt.annotate('y = {m}x + {c}'.format(m=round(m,3),c=round(b,3)),(1.50,6),color = "red")
    plot = plt.show()
    return plot

def ln_strength_vs_youngs(X,Y):
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, Y, 'x')
    plt.plot(X, m*X + b)
    plt.title('Natural log of Strength vs Youngs Modulus (MD and TD data)')
    plt.xlabel('ln(Strength)')
    plt.ylabel("ln(Young's Modulus)")
    plt.annotate('y = {m}x + {c}'.format(m=round(m,3),c=round(b,3)),(80,2),color = "red")
    plot = plt.show()
    return plot
    
def new_plot(X,Y):
    plt.plot(X, Y, 'x')
    plt.xlabel('strength')
    plt.ylabel("ln(Young's Modulus)")
    plot = plt.show()
    return plot
    
def ln_failure_vs_youngs(X,Y):
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, Y, 'x')
    plt.plot(X, m*X + b)
    plt.title('Natural log of Failure vs Youngs Modulus (MD and TD data)')
    plt.xlabel('ln(Failure)')
    plt.ylabel("ln(Young's Modulus)")
    plt.annotate('y = {m}x + {c}'.format(m=round(m,3),c=round(b,3)),(100,2),color = "red")
    plot = plt.show()
    return plot

#testing which model is more accurate
results = []
names= []
models = []
models.append(('LinReg', LinearRegression()))
models.append(('Lasso', Lasso()))
models.append(('Elastic', ElasticNet()))
models.append(('SVR', SVR()))
models.append(('Ridge',Ridge()))

def model_tester(X,Y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.25,random_state=0)
    for name, model in models:
        kfold = KFold(n_splits=2, random_state=0,shuffle=True)
        cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))   
       
scores = []
std_errors = []
def machine_learning(X,Y,model):
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
    print(scores_avg,"±",error_avg, "for {m}".format(m=model))
    scores_np = np.array(scores)
    plt.hist(scores_np, range = (scores_np.min(),scores_np.max()+0.10),rwidth =0.93, color = 'navy')
    plt.xlabel('R Squared Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of R squared scores ({m})'.format(m=model),fontweight = 'bold')
    plt.show()
    return scores_avg
def test_size(X,Y,model):
    sizes = np.arange(0.10,0.95,0.05)
    r_squa = []
    std_error = []
    for i in sizes:
            X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=i,random_state=0)
            model = model
            model.fit(X_train,Y_train)
            predictions = model.predict(X_validation)
            r_squa.append(r2_score(Y_validation,predictions))
    plt.plot(sizes,r_squa, 'x',color = 'red')
    plt.title('Test Size vs R squared',fontweight = 'bold')
    plt.xlabel('Test Size')
    plt.ylabel("R squared")
    plot = plt.show()
    
def cluster(X):
    model = KMeans(n_clusters=2)
    model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    labels = ['MD Data','TD Data']
    
    row_ix = where(yhat == clusters[0])
    row_ix2 =  where(yhat == clusters[1])
    pyplot.scatter(X[row_ix,0],X[row_ix,1],label = 'TD Data')
    pyplot.scatter(X[row_ix2,0],X[row_ix2,1], label = 'MD Data')
    plt.legend(loc='upper right',shadow = True, fontsize='x-small')
    plt.title('Clustering of Failure and Strength tensile data',fontweight = 'bold')
    plt.xlabel('Strength (MPa)')
    plt.ylabel("Failure (%)")
    
    pyplot.show()
        


    
    


def test_raw():
    test_r = [-2.1,0.24,0.62,0.62,0.64,0.75,0.77,0.76,0.76,0.77,0.78,0.76,0.76,-0.73,-2,-1.6]
    error = [1,0.69,0.26,0.26,0.30,0.17,0.12,0.13,0.13,0.12,0.12,0.10,0.12,1,1,1]
    sizes = np.arange(0.10,0.90,0.05)
    plt.errorbar(sizes,test_r,error,None,ls='none',ecolor = 'lightskyblue',capsize = 4)
    plt.plot(sizes,test_r, 'x',color = 'red')
    plt.title('Test Size vs R squared',fontweight = 'bold')
    plt.xlabel('Test Size')
    plt.ylabel("R squared")
    plt.show()



machine_learning(strength_failure, youngs_modulus, LinearRegression())
machine_learning(strength_failure ,youngs_modulus,Ridge())


test_size(strength_failure, youngs_modulus,Ridge())
test_raw()

cluster(strength_failure)



