import random
import numpy as np
import math
import matplotlib.pyplot as plt

seed = random.seed()
nums = 1000 # number of students
fract = 0.01 # fract initially immune
recovr = 0.2   # chance recovering / day
num_infect = 1 # number infected/day
numd = 40 # number of days
reps = 40
    



data = np.arange(1,numd+1,1)

sdata = np.arange(1,numd+1,1)

rdata = np.arange(1,numd+1,1)


remov = 0


for k in np.arange(1,reps+1,1):
    P = np.arange(1,nums+1,1)
    
    for i in np.arange(0,nums,1):
        P[i] = 1
    fn = math.trunc(fract*nums)
    n = 0
    while n < fn:
        rr = np.random.choice(np.arange(0,nums,1))
        ra = rr
        if P[ra] != 0:

            P[ra] = 0
            n = n + 1
    n = 0

    while n<1:
        rr = np.random.choice(np.arange(0,nums,1))
        ra = rr
        if P[ra] == 1:
            
            P[ra] = 2
            n = 1


    for j in np.arange(0,numd,1):
        for i in  np.arange(0,nums,1):


            if P[i] == 2:
             
                for ii in np.arange(1,num_infect+1,1):
                    ra = np.random.choice(np.arange(0,nums,1))
                    if P[ra] == 1:
                        P[ra] = 2
                        remov = remov + 1
        
        
            
            if np.random.rand() < recovr:
                ra = np.random.choice(np.arange(0,nums,1))
                if P[ra] == 2:
                    P[ra] = 0
        c = 0
        s = 0

        for i in np.arange(0,nums,1):
            if P[i] == 2:
                c = c+1
            if P[i] == 1:
                s = s+1
        data[j] = data[j] + c
        sdata[j] = sdata[j] + s


count = np.bincount(P)
print("Still infected on day 40 =",count[2])

num_infected = remov*(1/reps)
print("Totally infected overall =",num_infected)

data_days = data/reps
sdata_days = sdata/reps
rdata_days = rdata/(reps*numd*1000)
days= np.arange(1,numd+1)

#plot
plt.plot(days,data_days, label = "Infected")
plt.plot(days,sdata_days, label = "Susceptible", color = "red")
plt.title('Effect of virus on 1000 students over a 40 day period')
plt.xlabel('Day')
plt.ylabel("No. of Students")
plt.legend()
plt.show()
