# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:48:58 2023

@author: presvotscor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize,curve_fit
import math
import time
#from numpy.polynomial import chebyshev

class Models:
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        #inputs
        self.fn=fn # fréquence nominale du réseau électrique
        self.fs=fs # fréquence d'échantillonnage
        self.N=N # nombre d'échantillons
        self.verbose = verbose
        
        #constant 
        self.Ts=1/fs
        self.T=N/fs # durée d'une fenêtre
        self.Tn=1/fn # durée d'une période
        self.t=np.linspace(0,(N-1)*self.Ts,N)
        
        
        self.max_iter_model=30
        self.tol_models=1e-6
        #if (self.verbose):
        #    print("T",self.T)
        
    def norm(self,x,y):
        error = np.sum(np.square(y - x))
        #error = np.sum(np.abs(y - x))
        return error
 
class Model_sin(Models): 
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
    #    print("N",self.N)
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #    print("T",self.T)
    
    #"""
    def get_theta_sin_ini(self,y):
        a=np.std(y)*np.sqrt(2)

        index_max_y=list(y).index(np.max(y))
        delta=self.t[index_max_y]
        
        phi=-(delta*2*math.pi)/self.Tn 
        
        ### phi est ramené dans l'intervalle - pi pi
        while abs(phi)>np.pi:
            phi-=np.sign(phi)*2*np.pi
     

        if self.verbose:
            print("max y: {:.2f}".format(np.max(y)))
            print("index max y {}".format(index_max_y))
            print("delta t: {:.2f}".format(delta))
            print("phi: {:.2f}".format(phi))
     
        
        return [a,self.fn,phi]
    #"""    

    def get_model_sin(self,t,*theta):
        #print("theta",theta)
        return theta[0]*np.cos(2*math.pi*theta[1]*t+theta[2])

    
    def cost_function_sin(self,theta,y):
        x=self.get_model_sin(self.t,*theta)
        return self.norm(x,y)
        
    def get_theta_sin(self,y,m,w):
        #theta_ini=self.get_theta_sin_ini(y) # theta0
        #if self.verbose:
        #    print("theta ini", theta_ini)
            
        bounds = [(m[i]-w[i]/2, m[i]+w[i]/2) for i in range(3)]
        
        
        result = minimize(self.cost_function_sin, m, y, method='SLSQP', bounds=bounds)#options={'maxiter': self.maxiter}
        return [*result.x]

            
        #hat_theta, _ = curve_fit(self.get_model_sin,self.t,y,p0=theta_ini)
        #return hat_theta
        #print("[*result.x]",[*result.x])
        



"""
class Model_poly(Models):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        super().__init__(fn,fs,N,verbose) 
        
        self.t_poly=np.linspace(-1,1-1/self.N,self.N)

    
    
    def get_model_poly(self, t, *theta):
        model=chebyshev.chebval(self.t_poly, theta)
           
        return model
    

    
    def get_theta_poly(self, y,m,w, order):
        theta = chebyshev.chebfit(self.t_poly, y, order)
        #print(theta)
        return theta
"""



"""
class Model_poly(Models):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #    print("T",self.T)
        
        
        order=16
        self.chebyshev_term=np.zeros((order,self.N))
        
        tt=2 *self. t / self.T - 1
        for k in range(order):
            chebyshev_term =np.polynomial.chebyshev.Chebyshev([0] * k + [1])
            self.chebyshev_term[k]=chebyshev_term(tt)
            #print("chebyshev_term",chebyshev_term)
        #print(self.chebyshev_term)
      

    def get_theta_poly_ini(self, y, order):
       
        return [np.mean(y)] + [0.] * (order)
    
    
    
    
    def get_model_poly(self, t, *theta):
        order = len(theta) - 1
        
        model = np.zeros_like(t)
        
        for i in range(order+1):
            #chebyshev_term = np.polynomial.chebyshev.Chebyshev([0] * i + [1])  # Polynôme de Tchebychev d'ordre i
 
            #model += theta[i] * chebyshev_term(i)
            
            model += theta[i] * self.chebyshev_term[i]
           
        return model
    
    
    def cost_function_poly(self,theta,y):
        x=self.get_model_poly(self.t,*theta)
        return self.norm(x,y)
    
    def get_theta_poly(self,y,m,w,order):
        #theta_ini = self.get_theta_poly_ini(y, order)
       
        #if self.verbose:
        #    print("theta ini", theta_ini)
            
            
        bounds = [(m[i]-w[i]/2, m[i]+w[i]/2) for i in range(order+1)]

        result = minimize(self.cost_function_poly, m, y, method='SLSQP',tol=self.tol_models,options={'maxiter':self.max_iter_model},bounds=bounds)

 
        #hat_theta, _ = curve_fit(self.get_model_poly(self.t,*theta),self.t,y,p0=theta_ini)
        return [*result.x]
"""            


#"""
class Model_poly(Models):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #    print("T",self.T)
        
        self.t_cheby_p=np.linspace(-1,1-1/self.N,self.N)
        degree=16
        basis = [np.ones_like(self.t_cheby_p), self.t_cheby_p]
        for i in range(2, degree + 1):
            basis.append(2 * self.t_cheby_p * basis[i-1] - basis[i-2])
            
        #print(basis)
        #print(np.shape(basis))
        self.chebyshev_basis=np.vstack(basis).T
        #print(self.chebyshev_basis)
        #print(np.shape(self.chebyshev_basis))
    
    
    def get_model_poly(self, t, *theta):
        #print("len t",len(t))
        #print("len tch,",len(self.t_cheby_p),self.N)
        x_rec = np.polynomial.chebyshev.chebval(self.t_cheby_p, theta)

        #print("len xrec",len(x_rec))
        return x_rec
    
    
    
    def get_theta_poly(self, y,m,w, order):
        hat_theta, _, _, _ = np.linalg.lstsq(self.chebyshev_basis[:,:order+1], y, rcond=None)
        hat_theta=np.clip(hat_theta, [m[i]-w[i]/2 for i in range(len(m))], [m[i]+w[i]/2 for i in range(len(m))])
        #print("hat_theta",hat_theta,order)
        #hat_theta, _ = curve_fit(self.get_model_poly(self.t,*theta),self.t,y,p0=theta_ini)
        return hat_theta
    
#"""    
    
    
    
    
    
    
    


class Model_pred_samples(Models): 
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        #print("N",self.N)
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #print("T",self.T)
     

    def get_m_theta_pred_samples(self,N_p,eta,sigma,m,w):
        
        
        yp=np.array([0.75*np.cos(2*np.pi*self.fn*k*(1/self.fs)) for k in range(3*self.N)])+np.random.normal(0,sigma,3*self.N)
        X=self.get_X(yp[0:2*self.N],N_p,eta)
        m_theta_pred_samples=self.get_theta_pred_samples(X,yp[2*self.N:],m,w)
        return m_theta_pred_samples

    
    def get_model_pred_samples(self,X,*alpha):
        #print(np.size(X,1))
        #print(np.array(alpha))
        #print("alpha.reshape((np.size(X,1),1))",np.array(alpha).reshape((np.size(X,1),1)))
       
        #print(X)
        #print(np.shape(X))
        #print(np.shape(np.array(alpha).reshape((np.size(X,1),1))))
        
        x_rec=X @ np.array(alpha).reshape((np.size(X,1),1))
        x_rec=x_rec.reshape(self.N)
        #print(x_rec_test)        
        
        return x_rec    
            

    def get_X(self,y_p,N_p,eta):
        
        X=np.zeros((self.N,N_p))

        for i in range(self.N):
            A=np.array(y_p[self.N+i-eta-N_p+1:self.N+i-eta+1])
            #print(A)
            #print(A[::-1])
            X[i]=A[::-1]
        #print(np.linalg.norm(X, axis=0))
        #X_n=(X-np.mean(X,axis=0))/np.linalg.norm(X, axis=0) # np.linalg.norm(X, axis=1, keepdims=True)#
        #print(X_n)
        #print(x1_rec[(k-1)*N:k*N])
        return X
    
    
    def cost_function_pred_samples(self,alpha,y,X):
        x=self.get_model_pred_samples(X,*alpha)
        return self.norm(x,y)
    
    
    def get_theta_pred_samples(self,X,y,m,w):
        
        
        
        #print(X)
        # Calculer a=(A^T*A)^(-1)A^T*B
        #tpsa = time.perf_counter()
        #X+=np.random.normal(0,10**(-8),size=(self.N,np.size(X,1)))
        #hat_alpha = (np.linalg.inv(X.T @ X) @ X.T @ y.reshape((self.N,1)))
        #hat_alpha=hat_alpha.reshape(np.size(hat_alpha,0))
        #tpsb = time.perf_counter()
        #print("time 1",tpsb-tpsa)
        
        #print(np.linalg.inv(X.T @ X) @  X.T  @  X.T[0])
       
        """
        #tpsa = time.perf_counter()
        
        order=len(w)
        bounds = [(m[i]-w[i]/2, m[i]+w[i]/2) for i in range(order)]
        
        result = minimize(self.cost_function_pred_samples, m, args=(y,X),method='SLSQP',tol=self.tol_models,options={'maxiter':self.max_iter_model},bounds=bounds)
        return result.x 
        
        
        """
        hat_alpha, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        hat_alpha=hat_alpha.reshape(np.size(hat_alpha,0))
        #tpsb = time.perf_counter()
        #print("time 2",tpsb-tpsa)
        
        
        hat_alpha=np.clip(hat_alpha, [m[i]-w[i]/2 for i in range(len(m))], [m[i]+w[i]/2 for i in range(len(m))])
 
        
        #hat_alpha[-1]=1-np.sum(hat_alpha[0:N_p-1])
        #print(hat_alpha.reshape(N_p))
        return hat_alpha
        #"""



  
# Programme principal
if __name__ == "__main__":
    from Measures import get_snr,get_snr_l1
    from subsampling import dynamic_subsample
    verbose = False
    N=256
    fn=60
    fs=15384.6
    
    t=np.linspace(0,(N-1)/fs,N)
    
 
    
    model_sin=Model_sin(fn,fs,N,verbose)
    
    sigma=0.01 # écart type du bruit introduit dans le signal test
    
    
    
    
    
    """
    test model sinusoïdal
    """
    m=[0.75,fn,0]
    w=[0.5,0.2,2*np.pi]
    
    a=np.random.uniform(m[0]-w[0]/2,m[0]+w[0]/2)
    f=np.random.uniform(m[1]-w[1]/2,m[1]+w[1]/2)
    phi=np.random.uniform(m[2]-w[2]/2,m[2]+w[2]/2)
    
    theta=[0.6,f,phi]
    
    
    x_sin=model_sin.get_model_sin(t,*theta)+np.random.normal(0,sigma,N) 
    #x_sin[20]=20
    #x_sin[40]=15
    #x_sin[60]=20
    
    theta_sin_hat=model_sin.get_theta_sin(x_sin,m,w)
   
    x_sin_hat=model_sin.get_model_sin(t,*theta_sin_hat)
    
    #x_sin_ini=model_sin.get_model_sin(t,*m)#model_sin.get_model_sin(t,*model_sin.get_theta_sin_ini(x_sin))
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin,lw=2,label='x')
    #plt.plot(t,x_sin_ini,lw=2,label='x ini, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_ini)))
    plt.plot(t,x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle sinusoidal, theta=[{:.2f},{:.2f},{:.2f}]".format(*theta_sin_hat))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    """
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin-x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Erreur de reconstrucction")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    """
    
    
    
    
    
    
    
    
    
    """
    test polynôme d'ordre k
    """
    
    #"""
    order=3
    m=[0]*(order+1)
    w=[2]*(order+1)
    theta=[np.random.uniform(m[i]-w[i]/2,m[i]+w[i]/2) for i in range(order+1)]
  
    model_poly=Model_poly(fn,fs,N,verbose)
    
    
    x_poly=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)

    theta_poly_hat=model_poly.get_theta_poly(x_poly,m,w,order)
    x_poly_hat=model_poly.get_model_poly(t,*theta_poly_hat)
    
    #x_poly_ini=model_poly.get_model_poly(t,*model_poly.get_theta_poly_ini(x_poly,order))
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_poly,lw=2,label='x')
    #plt.plot(t,x_poly_ini,lw=2,label='x ini, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_ini)))
    plt.plot(t,x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle polynomial d'ordre {}, theta={}".format(order,[np.round(100*theta_poly_hat[i])/100 for i in range(order+1)]))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
   
    """
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_poly-x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Erreur de reconstruction poar le modèle polynomial d'ordre {}".format(order))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
    """
   
    
    
    #"""

    
   
    
   
    
    """
    test pred samples
    """
    
    
    eta=0
    N_p=4 # ordre du prédicteur
    

    pred=Model_pred_samples(fn,fs,N,verbose)
    
    model_sin=Model_sin(fn,fs,N,verbose)

    t_p=np.linspace(-(N*2)/fs,0-1/fs,N*2)
    t=np.linspace(0,(N-1)/fs,N)
    



    ########################################

    nb_test=100
    sigma=0.01

    alpha_L=np.zeros((nb_test,N_p))
    SNR=np.zeros(nb_test)
    #On créer la matrice X à partir des x_rec, on cherche à prédir les échantillons (x_{kN+1}-x_{(k+1)N})
    
    m=[0]*(N_p)
    w=[3]*(N_p)
    for i in range(nb_test):
        
    
        a=np.random.uniform(0.5,1)
        f=np.random.uniform(fn-0.5,fn+0.5)
        phi=np.random.uniform(-math.pi,math.pi)
        theta=[a,f,phi]
        
        a2=np.random.uniform(0,0.2)
        f2=3*f
        phi2=np.random.uniform(-math.pi,math.pi)
        theta2=[a2,f2,phi2]
  
        
        x_p=model_sin.get_model_sin(t_p,*theta)+model_sin.get_model_sin(t_p,*theta2)+np.random.normal(0,sigma,2*N) 
        x=model_sin.get_model_sin(t,*theta)+model_sin.get_model_sin(t,*theta2)+np.random.normal(0,sigma,N)     
        
        
        X=pred.get_X(x_p, N_p, eta)
        
        #m=pred.get_m_theta_pred_samples(N_p,eta,sigma)
        #m=[0]*N_p
        #w=[0.5]*N_p
        alpha_hat=pred.get_theta_pred_samples(X,x,m,w)
        #print(np.sum( alpha_hat))
        #print(alpha_L[i])
        #print(alpha_hat.reshape(N_p))
        alpha_L[i]=alpha_hat
        
        
        x_rec=pred.get_model_pred_samples(X,*alpha_hat)
        SNR[i]=get_snr(x, x_rec)
    
        if i==0:
            
            
            SNR1=get_snr(x, x_p[N:2*N])
            plt.figure(figsize=(15,8), dpi=150)
            plt.plot(t_p,x_p,lw=1,label="x_p")
            plt.plot(t,x,lw=1,label="x")
            plt.plot(t,x_p[N:],lw=1,label="x_rec_test, SNR={:.1f} dB".format(SNR1))
            plt.xlabel('t [s]')
            plt.ylabel('amplitude')
            plt.legend()
            plt.title("Superposition de la fenêtre précédente pour reconstruire x")
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
                
            plt.figure(figsize=(15,8), dpi=100)
            plt.plot(t_p,x_p,lw=1,label="xp")
            plt.plot(t,x,lw=1,label="x")
            plt.plot(t,x_rec,lw=1,label="x rec,  SNR={:.1f} dB".format(SNR[i]))
            plt.xlabel('t [s]')
            plt.ylabel('amplitude')
            plt.legend()
            plt.title("Reconstruction avec eta={}, Np={}".format(eta,N_p))
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
    
    print(N_p,eta,sigma,fn,fs,N)
    m_theta_pred_samples=pred.get_m_theta_pred_samples(N_p,eta,sigma,m,w)
    #print("m_theta_pred_samples",m_theta_pred_samples)
    for i in range(N_p):     
        plt.figure(figsize=(15,8), dpi=100)
        plt.hist(alpha_L[:,i],bins=int(nb_test/4),label="{}".format(i))
        plt.xlabel('amplitude')
        plt.ylabel('Nombre')
        plt.legend()
        plt.title("Reconstruction avec eta={}, Np={}, m={:.2f},  m_est={:.2f}, sigma={:.2f},".format(eta,N_p,np.mean(alpha_L[:,i]),m_theta_pred_samples[i],np.std(alpha_L[:,i])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()          
            
   
    
