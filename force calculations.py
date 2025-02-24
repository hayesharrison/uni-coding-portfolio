from scipy.constants import k
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate




def V(x):
    return (kf*(x**2)*(nm**2))/2 #nm means V(x) is in joules

N = 20000
deltax = 0.15 #nm
nm = 10**(-9)
T = 300.0
kT = k*T
kf = 10.0 #Newtons per Metre



Etot = 0.0
E2tot = 0.0 
xold = 0.0 #nanometres
Eold = V(xold)
Xtot = 0.0
X2tot = 0.0

for i in range(1,N):
    xnew = xold + np.random.uniform(-1,1)*deltax
    Enew = V(xnew)
    DeltaE = Enew - Eold
    if DeltaE <= 0.0:
        xold = xnew
        Eold = Enew   
    
    elif DeltaE > 0.0:
        if np.exp(-DeltaE/kT) > np.random.rand():
            xold=xnew
            Eold=Enew
        else:
            xold=xold
    Etot = Etot + Eold #changed this line
    E2tot = E2tot + Eold**2 #possible mistake here
    Xtot = Xtot + xold # add to total <X>
    X2tot = X2tot + xold**2 # add to total <X2>
    
Eav = (Etot/N)
E2av = (E2tot/N)
CV = (E2av-(Eav**2))/(k*T**2)
Xav = (Xtot/N)
X2av = (X2tot/N)    
error = np.sqrt(k*(T**2)*CV)/Eav
print("Average Energy =",Eav)
print("Average Squared Energy =", E2av)
print("Cv =",CV)
print("Average X position =",Xav )
print("Average X squared = ", X2av)

#Task 1)d)

def f(x):
    m = 9.11*10**(-31)
    return ((x**2)*np.exp(-(x**2)/(2*k*T)))

def f2(x):
    m = 9.11*10**(-31)
    return (np.exp(-(x**2)/(2*k*T)))

numerator = integrate.quad(f,-10**(-8),10**(-8))
denominator = integrate.quad(f2,-10**(-8),10**(-8))
print("\n" ,numerator)
print("\n" ,denominator)

print("\n" , 0.5*(numerator[0]/denominator[0]))


#Task 4

Nlist = []
errorlist = []
for i in range(2, 7):
    Nlist.append(pow(10, i))



for i in Nlist:
    errorlist.append(1/np.sqrt(i))




plt.plot(Nlist, errorlist)
plt.scatter(Nlist, errorlist)
plt.title("Effect of amount of measurements on the uncertainty in the average energy")
plt.xlabel("Fractional error in average energy")
plt.ylabel("Number of repeated measurements, N")
plt.show()
