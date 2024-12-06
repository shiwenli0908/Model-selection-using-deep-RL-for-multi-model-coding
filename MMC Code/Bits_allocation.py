# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:53:45 2023

@author: coren
"""


import numpy as np
from scipy.optimize import fsolve,minimize
import matplotlib.pyplot as plt

from Measures import get_snr,get_quality,get_mse,curve_tex

class Allocation:
    def __init__(self,verbose=False):
        self.verbose=verbose
        self.tol=1e-300
        self.max_iter=500
  
        
    def round_allocation(self,L,nx):
        
        
        root=np.array(L)
        #if nx<10:
        #    print("before",root,nx)
   
        root = np.maximum(L, 0)
                
                
        root=list(map(int, np.ceil(root)))
      
        if np.sum(root)>nx:
            i=-1
            while np.sum(root)!=nx:
                if root[i%(-len(L))]>0:
                    root[i%(-len(L))]-=1
                i-=1
        elif  np.sum(root)<nx:
            i=1
            while np.sum(root)!=nx:
                root[i%(len(L))]+=1
                i+=1
        
         
        return root
        
        
        
        """
        Arrondit les éléments d'un vecteur L de flottants pour obtenir un vecteur d'entiers positifs dont la somme est égale à bm,
        en minimisant l'erreur entre L et le vecteur arrondi.
        
        Args:
            L (array-like): Vecteur de flottants à arrondir.
            bm (float): Cible de la somme du vecteur arrondi.
            verbose (bool): Afficher des informations de débogage (par défaut False).
            
        Returns:
            array: Vecteur d'entiers positifs dont la somme est égale à bm.
        """
        
        """
        L=np.floor(L).astype(int)
        
        j=0
     
        while np.sum(L)!=nx:
            L[j%len(L)]+=1
            j+=1
           
        return L
        """ 
        
        bm=nx
        #"""
        #if self.verbose:
        #L=np.array(L)    
        #L=(L*nx)/sum(L)
        #print("L:", L, "Somme:", np.sum(L),nx)
            
        # Mettre les éléments négatifs à zéro
        """
        if np.min(L)<0:
            root=L-np.min(L)
        else :
            root=L
        """
        root=np.array(L)
        
        root = np.maximum(L, 0)
        
        # Réajuster l'allocation en fonction de la somme de root
        total_allocated = np.sum(root)
        if total_allocated != 0:
            root *= bm / total_allocated
        
        # Arrondir les éléments de root
        #root =np.round(root)#np.array([int(np.round(root[i])) for i in range(len(root))])
        root=list(map(int, np.round(root)))
        if self.verbose:
            print("root:", root, "Somme:", np.sum(root))
        
        # Vérification de si la cible bm est atteinte
        ecart = np.sum(root) - bm
        
        if ecart == 0:
            if self.verbose:
                print("Écart = 0, root:", root)
            return root
        
        if ecart > 0:  # Il faut enlever des bits pour atteindre la cible
            res = [root[i] - L[i] for i in range(len(root))]
            
            if self.verbose:
                print("Écart > 0, res:", res)
                
            while np.sum(root) != bm:
                index_max = res.index(max(res))
                if self.verbose:
                    print("Index max:", index_max)
                root[index_max] -= 1
                res[index_max] -= 1
            
            if self.verbose:
                print("Écart > 0, root:", root)
            
            return root
        
        if ecart < 0:  # Il faut ajouter des bits pour atteindre la cible
            res = [root[i] - L[i] for i in range(len(root))]
            
            if self.verbose:
                print("Écart < 0, res:", res)
                
            while np.sum(root) != bm:
                index_min = res.index(min(res))
                if self.verbose:
                    print("Index min:", index_min)
                root[index_min] += 1
                res[index_min] += 1
            
            if self.verbose:
                print("Écart < 0, root:", root)
            
            return root
     
        #"""
 
    
    
    

class Allocation_sin(Allocation):
    def __init__(self,fn,fs,N,max_bits,verbose=False):
        self.N=N
        self.fs=fs
        self.fn=fn
        self.NbyCycle=fs/fn
        self.max_bits=max_bits
        self.verbose = verbose
        #super().__init__() 
        Allocation.__init__(self,verbose)  
        
     

    def gamma0_sin(self,x,m,w):
        
        #a=0.75
        return  (w[0]**2*(2**(-2*x[0])) 
                             +  w[1]**2*(2**(-2*x[1]))*(m[0]**2)*((4*np.pi**2)/3)*(self.fn**(-2)) 
                             +  w[2]**2*(2**(-2*x[2]))*(m[0]**2))/(24)
                
    def gamma1_sin(self,x,m,w): 

        return self.gamma0_sin(x,m,w)*np.cos(2*np.pi/self.N) 
        #return (1-1/self.N)*(np.cos(2*np.pi/self.N))*(w[0]**2*(2**(-2*x[0])) 
        #                    +  w[1]**2*(2**(-2*x[1]))*(w[0]**2/12+m[0]**2)*((4*np.pi**2))*(self.fn**(-2))*(1/3+0.5/self.N) 
        #                     +  w[2]**2*(2**(-2*x[2]))*(w[0]**2/12+m[0]**2))/(24)

    def get_nx_sin(self,nx,m,w,dtype):
        
        """
    
        
        cons = ({'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]-nx} )
        
        bnds = ((0, self.max_bits), (0,self.max_bits),  (0,self.max_bits))
               
        x0 = [nx/3]*3
      
        res = minimize(self.gamma0_sin, x0,args=(m,w), method='SLSQP',bounds=bnds,
                       constraints=cons)
      
        root=res.x

        #print(res)
        #print("root reel",root,np.sum(root))
        if dtype=='int':    
            root=self.round_allocation(root, nx)    
        return root
        
        """
        Np=3
        pw=np.array([w[0]**2,(m[0]**2)*(4*np.pi**2/3)*(self.fn**(-2))*w[1]**2,(m[0]**2)*w[2]**2])
        root_theo=[0,0,0]
        for k in range(3):
            root_theo[k]=0.5*(-(1/Np)*np.sum(np.log2(pw))+2*nx/Np+np.log2(pw[k]))
        #print("root theo",root_theo,np.sum(root_theo))
        
        #print(res)
        #print("root reel",root,np.sum(root))
        if dtype=='int':    
            root_theo=self.round_allocation(root_theo, nx)    
        #else :
        #    for k in range(3):
        #        if root_theo[k]<0:
        #            root_theo[k]=0
                    
        return root_theo
       
    
    
    
    
    def get_all_nx_sin(self,nx_max,m_theta_sin,w_theta_sin,dtype):
        vect_allocation_sin=np.zeros((nx_max,3))
        for k in range(nx_max):
            vect_allocation_sin[k]=self.get_nx_sin(k,m_theta_sin,w_theta_sin,dtype)
        return vect_allocation_sin
            
    
    
    
    
    def MSE_theoritical_sin(self,x,m,w,gamma0_em,gamma1_em):
        #R=(np.sum(x[0:len(w)])+self.N*x[len(w)])/self.N
        #print((self.N/self.NbyCycle)**2)
        return ((gamma0_em**2-gamma1_em**2)/gamma0_em+(self.gamma0_sin(x[0:3],m,w)**2-self.gamma1_sin(x[0:3],m,w)**2)/self.gamma0_sin(x[0:3],m,w))*2**(-2*x[3])
        
    def get_nx_nr_constraint_bit_sin(self,m,w,error_model,btot,dtype='int'):
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(N-1)
        

        """
        #fun = lambda x: (2*(gamma0_em**2-1*gamma1_em**2)/(gamma0_em)+(N*self.fn/self.fs)**2*0.125*(self.gamma0_sin(x[0:3],m,w)**2-self.gamma1_sin(x[0:3],m,w)**2)/self.gamma0_sin(x[0:3],m,w))*2**(-2*x[3]/self.N)
        

        cons = ({'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]*self.N-btot})
        
        bnds = ((0, self.max_bits), (0, self.max_bits),  (0, self.max_bits), (0, 15))
               
        x0 = [8,8,8,2]
        res = minimize(self.MSE_theoritical_sin, x0,args=(m,w,gamma0_em,gamma1_em), method='SLSQP',bounds=bnds,
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter})
 
        root=res.x
  
        if dtype=='int':    
            root=self.round_allocation(root[0:3], np.sum(root[0:3]))    
        """
        root=[0,0,0,0]
        root_test=[0,0,0,0]
        MSE_memory=np.infty
        nx=0
        while nx<=btot:
            root_test[0:3]=self.get_nx_sin(nx,m,w,dtype)
            root_test[3]=(btot-nx)/self.N
            MSE_test=self.MSE_theoritical_sin(root_test,m,w,gamma0_em,gamma1_em)
            if MSE_test<MSE_memory:
                MSE_memory=MSE_test
                root=root_test
            else:
                break
            nx+=1
        #print("root",root)
            
        return root,self.MSE_theoritical_sin(root,m,w,gamma0_em,gamma1_em)  



         
    
    def get_nx_nr_constraint_MSE_sin(self,m,w,error_model,MSE_target,dtype='int'):

        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N-1)
        
        
        """
        fun = lambda x: x[0]+x[1]+x[2]+x[3]*self.N
        
        #MSE_tot=(2*(gamma0_em**2-1*gamma1_em**2)/(gamma0_em)+(N*self.fn/self.fs)**2*0.125*(self.gamma0_sin(x[0:3],m,w)**2-self.gamma1_sin(x[0:3],m,w)**2)/self.gamma0_sin(x[0:3],m,w))*2**(-2*x[3]/self.N)
        
        
        cons = ({'type': 'eq', 'fun': lambda x:  self.MSE_theoritical_sin(x,m,w,gamma0_em,gamma1_em)-MSE_target})
        
        bnds = ((0, self.max_bits), (0, self.max_bits),  (0, self.max_bits), (0, 5))
               
        x0 = [8,8,8,2]
        res = minimize(fun, x0, method='SLSQP',bounds=bnds,
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter})
        
        root=res.x
        #root[3]*=self.N
        if dtype=='int':    
            root=self.round_allocation(root, np.sum(root))    
        """
        
        root=[0,0,0,0]
        root_test=[0,0,0,0]
        btot_memory=np.infty
        nx=0
        while 1:
            root_test[0:3]=self.get_nx_sin(nx,m,w,dtype)
            gamma_0_q=self.gamma0_sin(root_test[0:3],m,w)
            gamma_1_q=self.gamma1_sin(root_test[0:3],m,w)
            
            MSE_test=(gamma0_em**2-gamma1_em**2)/gamma0_em+(gamma_0_q**2-gamma_1_q**2)/gamma_0_q
            
            #if MSE_test<=MSE_target:
            #    root_test[3]=0
            #else:
                
            root_test[3]=-0.5*np.log2(MSE_target/MSE_test)
            #print("MSE_target",MSE_target,"MSE_test",MSE_test,"nr=",root_test[3]*self.N)
            #print("MSE_target",MSE_test*2**(-2*root_test[3]))
            btot_test=nx+root_test[3]*self.N
            
            #print("btot",btot_test,nx)
            if btot_test<btot_memory:
                btot_memory=btot_test
                root=root_test
            else:
                #root[3]=np.max([0,root[3]])
                break
            nx+=1
        #print("root",root)  
        return root,self.MSE_theoritical_sin(root,m,w,gamma0_em,gamma1_em)  
    
    

    


class Allocation_None(Allocation):
    def __init__(self,fn,fs,N,max_bits,verbose=False):
        self.N=N
        self.fs=fs
        self.fn=fn
        self.max_bits=max_bits
        self.verbose = verbose
        #super().__init__() 
        Allocation.__init__(self,verbose)  
        
     
 
    def MSE_theoritical_none(self,x,gamma0_em,gamma1_em):

        #R=(np.sum(x[0:len(w)])+self.N*x[len(w)])/self.N
        #print((self.N/self.NbyCycle)**2)
        return ((gamma0_em**2-gamma1_em**2)/gamma0_em)*2**(-2*x)

    

    def get_nx_nr_constraint_MSE_none(self,error_model,MSE_target,dtype='int'):

        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N-1)
        
        


        
        MSE_test=(gamma0_em**2-gamma1_em**2)/gamma0_em
        
        #if MSE_test<=MSE_target:
        #    root_test[3]=0
        #else:
            
        root=-0.5*np.log2(MSE_target/MSE_test)
        #print("MSE_target",MSE_target,"MSE_test",MSE_test,"nr=",root_test[3]*self.N)
        #print("MSE_target",MSE_test*2**(-2*root_test[3]))
        #btot_test=nx+root*self.N
        

   
        #print("root",root)  
        return root,self.MSE_theoritical_none(root,gamma0_em,gamma1_em)      




















  
        
        
        
        

    
class Allocation_poly(Allocation):
    def __init__(self,fn,fs,N,max_bits,verbose=False):
        #inputs
        self.verbose = verbose
        self.N=N
        self.fn=fn
        self.fs=fs
        self.NbyCycle=fs/fn
        
        self.max_bits=max_bits
        
        mul=10
        N_tche=self.N*mul
        self.t_cheby=np.linspace(0,N_tche,N_tche)
   
        degree=15
        
        X=2*self.t_cheby/N_tche-1
        basis = [np.ones_like(X),
                 X,
                 2*X**2-1,
                 4*X**3-3*X,
                 8*X**4-8*X**2+1,
                 16*X**5-20*X**3+5*X,
                 32*X**6-48*X**4+18*X**2-1,
                 64*X**7-112*X**5+56*X**3-7*X,
                 128*X**8-256*X**6+160*X**4-32*X**2+1,
                 256*X**9-576*X**7+432*X**5-120*X**3+9*X,
                 512*X**10 - 1280*X**8 + 1120*X**6 - 400*X**4 + 50*X**2 - 1,
                 1024*X**11 - 2816*X**9 + 2816*X**7 - 1232*X**5 + 220*X**3 - 11*X,
                 2048*X**12 - 6144*X**10 + 6912*X**8 - 3584*X**6 + 840*X**4 - 72*X**2 + 1,
                 4096*X**13 - 13312*X**11 + 16640*X**9 - 9984*X**7 + 2912*X**5 - 364*X**3 + 13*X,
                 8192*X**14 - 28672*X**12 + 39424*X**10 - 26880*X**8 + 9408*X**6 - 1568*X**4 + 98*X**2 - 1,
                 16384*X**15 - 61440*X**13 + 92160*X**11 - 70400*X**9 + 28800*X**7 - 6048*X**5 + 560*X**3 - 15*X                         
                 ]
        
        X1=2*(self.t_cheby[mul:])/(N_tche)-1
        basis1 = [np.ones_like(X1),
                 X1,
                 2*X1**2-1,
                 4*X1**3-3*X1,
                 8*X1**4-8*X1**2+1,
                 16*X1**5-20*X1**3+5*X1,
                 32*X1**6-48*X1**4+18*X1**2-1,
                 64*X1**7-112*X1**5+56*X1**3-7*X1,
                 128*X1**8-256*X1**6+160*X1**4-32*X1**2+1,
                 256*X1**9-576*X1**7+432*X1**5-120*X1**3+9*X1,
                 512*X1**10 - 1280*X1**8 + 1120*X1**6 - 400*X1**4 + 50*X1**2 - 1,
                 1024*X1**11 - 2816*X1**9 + 2816*X1**7 - 1232*X1**5 + 220*X1**3 - 11*X1,
                 2048*X1**12 - 6144*X1**10 + 6912*X1**8 - 3584*X1**6 + 840*X1**4 - 72*X1**2 + 1,
                 4096*X1**13 - 13312*X1**11 + 16640*X1**9 - 9984*X1**7 + 2912*X1**5 - 364*X1**3 + 13*X1,
                 8192*X1**14 - 28672*X1**12 + 39424*X1**10 - 26880*X1**8 + 9408*X1**6 - 1568*X1**4 + 98*X1**2 - 1,
                 16384*X1**15 - 61440*X1**13 + 92160*X1**11 - 70400*X1**9 + 28800*X1**7 - 6048*X1**5 + 560*X1**3 - 15*X1                 
                 ] 
        
        
        
        self.c=[]
        self.c1=[]
        
        for i in range(0, degree + 1):

            self.c.append(np.mean(basis[i]**2))
            self.c1.append(np.sum(basis[i][0:N_tche-mul]*basis1[i])/(N_tche))
        
                               
        test_coefs=0
        if test_coefs:
            
            #♠coefs théorique
            delta=1/self.N
            self.c_=[1,1/3,7/15,17/35,31/63,49/99,71/143,97/195,127/255,161/323]
            self.c1_=[1,
                      1/3,
                      (7-40*delta**2)/15,
                      (17-336*delta**2)/35,
                      (155-6816*delta**2)/315,
                      (343-27280*delta**2)/693,
                      (7455-949416*delta**2)/15015,
                      (3201-602336*delta**2)/6435,
                      (381381-100154752*delta**2)/765765,
                      (805805-283006368*delta**2)/1616615]
            
            
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(self.c_[:]),label='c diff')
            plt.plot(np.log(self.c1_[:]),label='c1 diff')
            plt.plot(np.log(self.c[:]),"*",label='c discret')
            plt.plot(np.log(self.c1[:]),"*",label='c1 discret')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
        
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(np.array(self.c_[:])-np.array(self.c1_[:])),label='c-c1 diff')
            plt.plot(np.log(np.array(self.c[:])-np.array(self.c1[:])),"-*",label='c-c1 discret')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
        
        
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.array(self.c_[0:9])-np.array(self.c[0:9]),label=' c diff - c discret ')
            plt.plot(np.array(self.c1_[0:9])-np.array(self.c1[0:9]),label=' c1 diff - c1 discret')
            #plt.plot(self.c[1:],label='c1 à partir de c')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
     
        
    

        Allocation.__init__(self,verbose) 


    def gamma0_poly(self,x,w):
        return np.sum([self.c[i]*w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/(12)
    
    def gamma1_poly(self,x,w):
        #return (1-1/self.N)*np.sum([self.c[i]*w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/12
        
        return np.sum([self.c1[i]*w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/12
    
    
    
    
    

    
    def get_nx_poly(self,nx,w,dtype='int'):
        """
        Np=len(w)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[0:Np])-nx})
        
        bnds =  tuple([(0, self.max_bits)]*(Np))
    
        x0 = [nx/Np]*(Np)

        res = minimize(self.gamma0_poly, x0, args=(w), method='SLSQP',bounds=bnds,
                       constraints=cons)
        #print(res)
        root=res.x
       
        
        #root=[nx/Np]*(Np)
        if dtype=='int': 
            root=self.round_allocation(root[0:Np], nx)  

        return root
       
        """
        
        Np=len(w)
        pw=np.array([self.c[k]*w[k]**2 for k in range(Np)])
        root_theo=[0]*Np
        for k in range(Np):
            root_theo[k]=0.5*(-(1/Np)*np.sum(np.log2(pw))+2*nx/Np+np.log2(pw[k]))
        #print("root theo",root_theo,np.sum(root_theo))
        
        #print(res)
        #print("root reel",root,np.sum(root))
        if dtype=='int': 
            #print("-------------------")
            #print(root_theo,nx,sum(root_theo))
            root_theo=self.round_allocation(root_theo, nx)    
        #else :
            #for k in range(Np):
            #    if root_theo[k]<0:
            #        root_theo[k]=0
                    
        #print(root_theo,nx)
        return root_theo
            
    
    
    def get_all_nx_poly(self,nx_max,w_theta_poly,dtype='int'):
        vect_allocation_poly=np.zeros((nx_max,len(w_theta_poly)))
        for k in range(nx_max):
            vect_allocation_poly=self.get_nx_poly(k,w_theta_poly,dtype)

        return vect_allocation_poly
            



    
    def MSE_theoritical_poly(self,x,w,gamma0_em,gamma1_em):
        return ((gamma0_em**2-gamma1_em**2)/gamma0_em+(self.gamma0_poly(x[0:len(w)],w)**2-self.gamma1_poly(x[0:len(w)],w)**2)/self.gamma0_poly(x[0:len(w)],w))*2**(-2*x[-1])
        
        
    
    
    
    
    
    
    
    
    def get_nx_nr_constraint_bit_poly(self,w,error_model,btot,dtype='int'):
        """
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(N-1)
        

        Np=len(w)
 
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[0:Np])+x[Np]*self.N-btot})
        
        bnds = tuple([(0, self.max_bits)]*Np+[(0,8)])
               
        x0 = [6]*Np +[(self.N-6*Np)/self.N]

        res = minimize(self.MSE_theoritical_poly, x0,args=(w,gamma0_em,gamma1_em), method='SLSQP',bounds=bnds,
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter})
        #print(res)
        root=res.x
  
        if dtype=='int':    
            root=self.round_allocation(root[0:Np], np.sum(root[0:Np]))    
   
        #efzefzef
        return root,self.MSE_theoritical_poly(root,w,gamma0_em,gamma1_em) 
        """
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(N-1)
        
       
 
        Np=len(w)
        root=[0]*(Np+1)
        root_test=[0]*(Np+1)
        MSE_memory=np.infty
        nx=0
        while nx<=btot:
            root_test[0:Np]=self.get_nx_poly(nx,w,dtype)
            root_test[Np]=(btot-nx)/self.N
            MSE_test=self.MSE_theoritical_poly(root_test,w,gamma0_em,gamma1_em)
            if MSE_test<MSE_memory:
                MSE_memory=MSE_test
                root=root_test
            else:
                break
            nx+=1
        return root,self.MSE_theoritical_poly(root,w,gamma0_em,gamma1_em)  
        #print("root",root)
    
    
    
    
    
    def get_nx_nr_constraint_MSE_poly(self,w,error_model,MSE_target,dtype='int'):
        """
        Np=len(w)
        fun = lambda x: np.sum(x[0:Np])+x[Np]*self.N
        
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N-1)
        

   
        
        cons = ({'type': 'eq', 'fun': lambda x:  MSE_target-self.MSE_theoritical_poly(x,w,gamma0_em,gamma1_em)})
        
        bnds =  tuple([(0, self.max_bits)]*Np+[(0,5)])
   
    
        x0 = [12]*Np +[(self.N-12*Np)/self.N]
        
        
        res = minimize(fun, x0,bounds=bnds,method='SLSQP',
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter})
        
        root=res.x
        
        print(res)
        #root[3]*=self.N
        if dtype=='int':    
            root=self.round_allocation(root, np.sum(root))    
        
       
        return root,self.MSE_theoritical_poly(root,w,gamma0_em,gamma1_em)  
        """

        Np=len(w)
       
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N-1)
        

       
        root=[0]*(Np+1)
        root_test=[0]*(Np+1)
        btot_memory=np.infty
        nx=0
        while 1:
            root_test[0:Np]=self.get_nx_poly(nx,w,dtype)
            gamma_0_q=self.gamma0_poly(root_test[0:Np],w)
            gamma_1_q=self.gamma1_poly(root_test[0:Np],w)
            
            MSE_test=(gamma0_em**2-gamma1_em**2)/gamma0_em+(gamma_0_q**2-gamma_1_q**2)/gamma_0_q
            root_test[Np]=-0.5*np.log2(MSE_target/MSE_test)
            #print("MSE_target",MSE_target,"MSE_test",MSE_test,"nr=",root_test[3]*self.N)
            #print("MSE_target",MSE_test*2**(-2*root_test[3]))
            btot_test=nx+root_test[Np]*self.N
            
            #print("btot",btot_test,nx)
            if btot_test<btot_memory:
                btot_memory=btot_test
                root=root_test
            else:
                break
            nx+=1
        return root,self.MSE_theoritical_poly(root,w,gamma0_em,gamma1_em)  

        

  

    
    
    
    
    
    
    


class Allocation_pred_samples(Allocation):
    def __init__(self,fn,fs,N,max_bits,verbose=False):
        #inputs
        self.N=N
        self.verbose = verbose
        self.fn=fn
        self.fs=fs
        self.NbyCycle=fs/fn
        
        self.method="SLSQP"
        self.max_bits=max_bits
        Allocation.__init__(self,verbose) 
        #super().__init__() 
        
    def gamma0_pred_samples(self,x,w,gamma0_q):
  
        return gamma0_q*np.sum([w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/(12)

    def gamma1_pred_samples(self,x,w,gamma0_q,gamma1_q):
            #sign=np.sign(gamma0_q-gamma1_q)
            return (1-0/self.N)*gamma1_q*np.sum([w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/(12)# 
            #return (1-1/self.N)*gamma0_q*np.sum([w[i]**2*(2**(-2*x[i])) for i in range(len(w))])/(12)# 
                          
        
    def get_nx_pred_samples(self,nx,w,eta,dtype='int'):
        
        
        """
        Np=len(w)
        
        #c=np.ones(Np)#[np.sum(xp[self.N-eta-k:2*self.N-eta-k]**2) for k in range(Np)]
        
        gamma0_q=1#np.mean(xp[self.N:2*self.N]**2)
        #c=[np.sum(xp[self.N-eta-k:2*self.N-eta-k]**2)/self.N for k in range(len(w))]
        
       
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[0:Np])-nx})
        
        bnds =  tuple([(0, self.max_bits)]*Np)
   
    
        x0 = [nx/Np]*(Np) 

        res = minimize(self.gamma0_pred_samples, x0, args=(w,gamma0_q),method='SLSQP',bounds=bnds,
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter})
        #print(res)
        root=res.x
      
        
        
        #root= [nx/Np]*(Np) 
        if dtype=='int':    
            root=self.round_allocation(root, nx)    
        
        return root
        """
        Np=len(w)
        pw=np.array([w[k]**2 for k in range(Np)])
        root_theo=[0]*Np
        for k in range(Np):
            root_theo[k]=0.5*(-(1/Np)*np.sum(np.log2(pw))+2*nx/Np+np.log2(pw[k]))
        #print("root theo",root_theo,np.sum(root_theo))
        
        #print(res)
        #print("root reel",root,np.sum(root))
        if dtype=='int':    
            root_theo=self.round_allocation(root_theo, nx)    
        else :
            for k in range(Np):
                if root_theo[k]<0:
                    root_theo[k]=0
                    
        #print(root_theo,nx)
        return root_theo
                    

    def get_all_nx_pred_samples(self,nx_max,w,eta,dtype='int'):
        vect_allocation=np.zeros((nx_max,len(w)))
        for k in range(nx_max):
            vect_allocation[k]=self.get_nx_pred_samples(k,w,eta,dtype)
        return vect_allocation   
    
    
    
    
    
    def MSE_theoritical_pred_samples(self,x,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q):
   
       # btot=np.sum(x[0:len(w)])+self.NbyCycle*x[len(w)]
        return ((gamma0_em**2-gamma1_em**2)/gamma0_em+(self.gamma0_pred_samples(x[0:len(w)],w,gamma0_q)**2-self.gamma1_pred_samples(x[0:len(w)],w,gamma0_q,gamma1_q)**2)/self.gamma0_pred_samples(x[0:len(w)],w,gamma0_q))*2**(-2*x[-1])
        
        
    #"""
    def get_nx_nr_constraint_bit_pred_samples(self,w,eta,xp,error_model,btot,dtype='int'):
        """
        Np=len(w)
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(N-1)
        #gamma1_em=np.min([np.mean(error_model[0:self.N-1]*error_model[1:self.N]),(1-1/N)*gamma0_em])
        
        gamma0_q=np.mean(xp[self.N:2*self.N]**2)
        gamma1_q=np.sum(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N])/(N-1)
        #gamma1_q=np.min([np.mean(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N]),(1-1/N)*gamma0_q])
       
 
        
        cons = ({'type': 'eq', 'fun': lambda x: (np.sum(x[0:Np])+x[Np]*self.N-btot)/btot})
        
        bnds = tuple([(0, self.max_bits)]*Np+[(0,15)])
               
        x0 = [20/btot]*Np +[(btot-Np*20)/(self.N*btot)]
        
        def fun(x,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q):
            return 50*np.log10(((gamma0_em**2-gamma1_em**2)/gamma0_em+(self.gamma0_pred_samples(x[0:len(w)],w,gamma0_q)**2-self.gamma1_pred_samples(x[0:len(w)],w,gamma0_q,gamma1_q)**2)/self.gamma0_pred_samples(x[0:len(w)],w,gamma0_q))*2**(-2*x[-1]))
       
        res = minimize(fun, x0,args=(w,gamma0_em,gamma1_em,gamma0_q,gamma1_q), 
                       method=self.method,bounds=bnds,
                       constraints=cons)
        print("get_nx_nr_constraint_bit_pred_samples",res)
        root=res.x
  
        if dtype=='int':    
            root=self.round_allocation(root[0:Np], np.sum(root[0:Np]))    
   
        #efzefzef
        print("root Np",root[0:Np]*btot,"Rr=",root[-1]*self.N*btot,fun(root,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q) )
        return root,self.MSE_theoritical_pred_samples(root,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q)  
        """
    
    
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N)
        #gamma1_em=np.min([np.mean(error_model[0:self.N-1]*error_model[1:self.N]),(1-1/N)*gamma0_em])
        
        gamma0_q=np.mean(xp[self.N:2*self.N]**2)
        gamma1_q=np.sum(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N])/(self.N)
        #gamma1_q=np.min([np.mean(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N]),(1-1/N)*gamma0_q])
       
 
        Np=len(w)
        root=[0]*(Np+1)
        root_test=[0]*(Np+1)
        MSE_memory=np.infty
        nx=0
        while nx<=btot:
            root_test[0:Np]=self.get_nx_pred_samples(nx,w,eta,dtype)
            root_test[Np]=(btot-nx)/self.N
            MSE_test=self.MSE_theoritical_pred_samples(root_test,w,gamma0_em,gamma1_em, gamma0_q, gamma1_q)
            if MSE_test<MSE_memory:
                MSE_memory=MSE_test
                root=root_test
            else:
                break
            nx+=1
        return root,self.MSE_theoritical_pred_samples(root,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q)  
        #print("root",root)



    #"""    
        
    def get_nx_nr_constraint_MSE_pred_samples(self,w,eta,xp,error_model,MSE_target,dtype='int'):
        """
        Np=len(w)
        fun = lambda x: np.sum(x[0:Np])+x[Np]*self.N
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(N-1)
        #gamma1_em=np.min([np.mean(error_model[0:self.N-1]*error_model[1:self.N]),(1-1/N)*gamma0_em])
        
        gamma0_q=np.mean(xp[self.N:2*self.N]**2)
        gamma1_q=np.sum(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N])/(N-1)
        #gamma1_q=np.min([np.mean(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N]),(1-1/N)*gamma0_q])
       
        cons = ({'type': 'eq', 'fun': lambda x:  self.MSE_theoritical_pred_samples(x,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q)-MSE_target})
        
        

        
        bnds = tuple([(0, self.max_bits)]*Np+[(0,15)])
               
        x0 = [8]*Np +[0.5]

        res = minimize(fun, x0,
                       method=self.method,bounds=bnds,
                       constraints=cons,tol= self.tol,options={'maxiter':self.max_iter*100})
        
        print(res)
        root=res.x
  
        if dtype=='int':    
            root=self.round_allocation(root[0:Np], np.sum(root[0:Np]))    
   
        #efzefzef
        return root,self.MSE_theoritical_pred_samples(root,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q)         
        """
        
        Np=len(w)
       
        
        gamma0_em=np.mean(error_model**2)
        gamma1_em=np.sum(error_model[0:self.N-1]*error_model[1:self.N])/(self.N)
        #gamma1_em=np.min([np.mean(error_model[0:self.N-1]*error_model[1:self.N]),(1-1/N)*gamma0_em])
        
        gamma0_q=np.mean(xp[self.N:2*self.N]**2)
        gamma1_q=np.sum(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N])/(self.N)
        #gamma1_q=np.min([np.mean(xp[self.N:2*self.N-1]*xp[self.N+1:2*self.N]),(1-1/N)*gamma0_q])
       
        root=[0]*(Np+1)
        root_test=[0]*(Np+1)
        btot_memory=np.infty
        nx=0
        while 1:
            root_test[0:Np]=self.get_nx_pred_samples(nx,w,eta,dtype)
            gamma_0_q=self.gamma0_pred_samples(root_test[0:Np],w,gamma0_q)
            gamma_1_q=self.gamma1_pred_samples(root_test[0:Np],w,gamma0_q,gamma1_q)
            
            MSE_test=(gamma0_em**2-gamma1_em**2)/gamma0_em+(gamma_0_q**2-gamma_1_q**2)/gamma_0_q
            root_test[Np]=-0.5*np.log2(MSE_target/MSE_test)
            #print("MSE_target",MSE_target,"MSE_test",MSE_test,"nr=",root_test[3]*self.N)
            #print("MSE_target",MSE_test*2**(-2*root_test[3]))
            btot_test=nx+root_test[Np]*self.N
            
            #print("btot",btot_test,nx)
            if btot_test<btot_memory:
                btot_memory=btot_test
                root=root_test
            else:
                break
            nx+=1
        return root,self.MSE_theoritical_pred_samples(root,w,gamma0_em,gamma1_em,gamma0_q,gamma1_q)  

        


        



        
        
     
    
'''
from Models import Model_poly
def generate_corelated_noise(amplitude,sigma,order_pred_poly,N):
    model_poly=Model_poly(fn,fs,N,verbose=False)
    
    
    w_theta_poly=[amplitude]*(order_pred_poly+1)
    theta_poly=[0]*(order_pred_poly+1)
    for i in range(order_pred_poly+1):
        theta_poly[i]=np.random.uniform(-w_theta_poly[i]/2,+w_theta_poly[i]/2)
    
    return model_poly.get_model_poly(t, *theta_poly)+np.random.normal(0,sigma,N)
'''
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf



def generate_corelated_noise(ar,ma,N,sigma,verbose):
    #yt=0.5 yt-1+0.2epst-1+epst
    #ar=[1,-0.5]
    #ma=[1,0.2]
    
    #test 
    #ar=[1,-0.8,-0.1]
    #ma=[1,0.4]
    #N=1024
    y = arma_generate_sample(ar,ma,N,sigma)
    #y-=np.mean(y)

    
    autocorr=np.correlate(y, y, mode='full')
    autocorr=autocorr[N-1:]/N

    if verbose:
        
        # instantiate model objet
        model=ARIMA(y,order=(len(ar)-1,0,len(ma)-2))
        
        # fit 
        results=model.fit()
        print(results.summary())
        
        
        plt.figure(figsize=(8, 4))
        plt.plot(y)
        plt.title("Série temporelle du bruit corrélé avec ARMA({},{})".format(len(ar)-1,len(ma)-1))
        plt.xlabel("Temps")
        plt.ylabel("Valeurs")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        # Tracer les autocorrélations
        
        plt.figure(figsize=(8, 4))
        plot_acf(y, lags=N-1)
        plt.title("Autocorrélation de la série temporelle")
        plt.xlabel("Lags")
        plt.ylabel("Autocorrélation")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        
        
        
        

        
        
        # Normalisation de l'autocorrélation

            
        max_=np.max(autocorr)
            
        # Tracé de l'autocorrélation
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(autocorr/max_,'-*')
        plt.title('Autocorrelation, MSE ={:.1f} dB, gamma0={:.1f} dB, gamma1={:.1f} dB'.format(10*np.log10(get_mse(y,np.zeros(N))),10*np.log10(autocorr[0]),10*np.log10(autocorr[1])))
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
    return y,autocorr

test_generate_noise=0
if test_generate_noise:
    ar=[1,-0.6]
    ma=[1,0]
    N=128*100
    sigma=1
    y,auto=generate_corelated_noise(ar,ma,N,sigma,True)
    
    
    ntest=20
    gamma0=np.zeros(ntest)
    gamma1=np.zeros(ntest)
    gamma0_gamma1=np.zeros(ntest)
    for k in range(ntest):
        y,auto=generate_corelated_noise(ar,ma,N,sigma,False)
        gamma0[k]=auto[0]
        gamma1[k]=auto[1]
        gamma0_gamma1[k]=auto[0]/auto[1]
        
    print("mean gamma0 = {:.3f}".format(np.mean(gamma0)))
    print("mean gamma1 = {:.3f}".format(np.mean(gamma1)))
    print("mean gamma0 / gamma1 = {:.3f}".format(np.mean(gamma0_gamma1)))
   




def autocorrelation(x,verbose):
    """
    Calculate and plot the autocorrelation of a given list of data.
    
    Parameters:
        data (list or numpy array): List of data points
        
    Returns:
        None
    """
    

    # Calcul de l'autocorrélation pour différents décalages
    
    mean_x=0*np.mean(x)
    N=len(x)
    autocorr=np.correlate(x-mean_x, x-mean_x, mode='full')
    autocorr=autocorr[N-1:]/N
    
    
    # Normalisation de l'autocorrélation
   
    
    
    if verbose:
        
        max_=np.max(autocorr)
        
        # Tracé de l'autocorrélation
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(autocorr/max_,'-*')
        plt.title('Autocorrelation, MSE ={:.1f} dB, gamma0={:.1f} dB, gamma1={:.1f} dB'.format(10*np.log10(get_mse(x,np.zeros(N))),10*np.log10(autocorr[0]),10*np.log10(autocorr[1])))
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
    
    return autocorr



# Programme principal
if __name__ == "__main__":
    from Models import Model_sin,Model_poly,Model_pred_samples
    
    
    #from codage_model import Model_Encoder
    from Quantization import Quantizer
    from Normalize import normalize
    from codage_residu import Residual_Encoder
    
    
    
    from get_test_signal import get_RTE_signal,get_EPRI_signal


    v1,v2,v3,i1,i2,i3=get_RTE_signal()
    v1=v1[23:]

    
    
    ################################################## définition corrélation
    verbose = False
    R=1
    N=int(128) #256
    fn=50 #60#
    fs=int(6400) #15384.6#
    max_bits=60#int(N*R) # 
    dtype='float'


    t=np.linspace(0,(N-1)/fs,N)
    
    
    ################################################ True si modèle  
    sin=False
    poly=True

    order=4
    pred_samples=False
    order_pred=2
    eta=0
    bmax=int(R*N)
    factor_scale=2
    ###################################################################### test allocation de bits model sin
    model_sin =Model_sin(fn,fs,N,verbose)
    allocation_sin = Allocation_sin(fn,fs,N,max_bits,verbose)
    
    
    model_poly =Model_poly(fn,fs,N,verbose)
    allocation_poly = Allocation_poly(fn,fs,N,max_bits,verbose)
  

    model_pred_samples=Model_pred_samples(fn,fs,N,verbose)
    allocation_pred_samples= Allocation_pred_samples(fn,fs,N,max_bits)
  
    quantizer=Quantizer(verbose) 
    
    l=Residual_Encoder(N,factor_scale)


    ########################################### Initialisation des listes MSE
    real=True
    
    
    fen=20
    
    for fen in range(fen,fen+1):
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t,v1[fen*N:(fen+1)*N],lw=2,label=r'$\mathbf{{x}}, fen={}$'.format(fen))
        plt.xlabel('t (s)')
        plt.ylabel('amplitude')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
 
 

    nb_signal=10
    if real :
        nb_signal=1
    
    ntot=int(R*N)
    ktot=1.
    nx_max=np.min([max_bits,ntot])
    #if real :
    #    nx_max=ntot
    
    
    
    ############ bruit corrélé ARMA aditioné au modèle 

    ########################################## test noise 

    phi=1.7
    phi2=-0.8
    phi_arma = [1,-phi,-phi2]  # Coefficients autoregressifs
    phi_arma = [1,-0.9]  # Coefficients autoregressifs
    sigma=1/(1000) #2.092*10**(-6)
    theta_arma = [1,0]  # Coefficients de la moyenne mobile
    y,auto=generate_corelated_noise(phi_arma,theta_arma,N,sigma,False)
    
 
    
    ###### erreurs entre le signal et le modèle 
    MSE_model=np.zeros(nb_signal)
    
    
    ###### erreurs entre le modèle et le modèle tilde
    MSE_model_q=np.zeros(nx_max)
    MSE_model_q_opt=np.zeros(nx_max)
    MSE_reel_q=np.zeros((nb_signal,nx_max))
    MSE_reel_q_opt=np.zeros((nb_signal,nx_max))
    
    
    ###### erreurs entre le residu et le résidu rec
    MSE_model_r=np.zeros((nb_signal,nx_max))
    MSE_model_r_opt=np.zeros((nb_signal,nx_max))
    MSE_reel_r=np.zeros((nb_signal,nx_max))
    MSE_reel_r_opt=np.zeros((nb_signal,nx_max))
    
    
    ####### erreur entre le signal et le modèle tilde
    #MSE_model_qm=np.zeros(nx_max)
    #MSE_model_qm_opt=np.zeros(nx_max)
    MSE_reel_qm=np.zeros((nb_signal,nx_max))
    MSE_reel_qm_opt=np.zeros((nb_signal,nx_max))    
    
    
    ####### erreur total de reconstruction
    MSE_model_tot=np.zeros((nb_signal,nx_max))
    MSE_model_tot_opt=np.zeros((nb_signal,nx_max))
    MSE_reel_tot=np.zeros((nb_signal,nx_max))
    MSE_reel_tot_opt=np.zeros((nb_signal,nx_max))
    
    
    
    al_opt_float=np.zeros((nx_max,3))
    al_opt_int=np.zeros((nx_max,3))
    
    al_float=np.zeros((nx_max,3))
    al_int=np.zeros((nx_max,3))
    
    gamma0_reel=np.zeros((nb_signal,nx_max))
    gamma0_reel_opt=np.zeros((nb_signal,nx_max))
    gamma0_model_opt=np.zeros((nx_max))
    gamma0_model=np.zeros((nx_max))
    
    gamma1_reel_opt=np.zeros((nb_signal,nx_max))
    gamma1_model_opt=np.zeros((nx_max))
    gamma1_reel=np.zeros((nb_signal,nx_max))
    gamma1_model=np.zeros((nx_max))
    
    
    scalar_product_m=np.zeros((nb_signal,nx_max))
    scalar_product_mq=np.zeros((nb_signal,nx_max))
    scalar_product_q=np.zeros((nb_signal,nx_max))
    
    
    if sin :
        m_theta_sin=[0.75,fn,0]
        w_theta_sin=[0.5,0.2,2*np.pi]
        
        for nx in range(nx_max):
            al_opt_float[nx]=allocation_sin.get_nx_sin(nx, m_theta_sin, w_theta_sin, dtype="float")
            al_opt_int[nx]=allocation_sin.round_allocation(al_opt_float[nx], nx)
    
            MSE_model_q_opt[nx]=allocation_sin.gamma0_sin(al_opt_float[nx],m_theta_sin,w_theta_sin)
            
            gamma0_model_opt[nx]=MSE_model_q_opt[nx]
            
            
            al_float[nx]=[nx/3]*3
            al_int[nx]=allocation_sin.round_allocation(al_float[nx], nx)
            #print("al sin float=[{:.2f},{:.2f},{:.2f}],  nx={:.2f}, nx target={}".format(*al_float,np.sum(al_sin_opt_float),nx))
            #print("al sin int=[{:.0f},{:.0f},{:.0f}]".format(*al_int))
            MSE_model_q[nx]=allocation_sin.gamma0_sin(al_float[nx],m_theta_sin,w_theta_sin)
            gamma0_model[nx]=MSE_model_q[nx]    
                
            
            
            gamma1_model_opt[nx]=allocation_sin.gamma1_sin(al_opt_float[nx],m_theta_sin,w_theta_sin)
            gamma1_model[nx]=allocation_sin.gamma1_sin(al_float[nx],m_theta_sin,w_theta_sin)
            
        for w in range(nb_signal):
            
            
            a=np.random.uniform(m_theta_sin[0]-w_theta_sin[0]/2,m_theta_sin[0]+w_theta_sin[0]/2)
            f=np.random.uniform(m_theta_sin[1]-w_theta_sin[1]/2,m_theta_sin[1]+w_theta_sin[1]/2)
            phi=np.random.uniform(m_theta_sin[2]-w_theta_sin[2]/2,m_theta_sin[2]+w_theta_sin[2]/2)
            
            theta_sin=[a,f,phi]
    
            
    
            signal_noise,_=generate_corelated_noise(phi_arma,theta_arma,N,sigma,False)
            signal_sin_clean=model_sin.get_model_sin(t,*theta_sin)
            
            if real:
                    
                
                signal_sin,_=normalize(v1[fen*N:fen*N+N])
            else :
                signal_sin=signal_sin_clean+signal_noise
            
            theta_sin_hat=model_sin.get_theta_sin(signal_sin,m_theta_sin,w_theta_sin)
            signal_sin_hat=model_sin.get_model_sin(t, *theta_sin_hat)
            
            if real :
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,signal_sin,lw=2,label=r'$\mathbf{x}$')
                plt.plot(t,signal_sin_hat,lw=2,label=r'$\mathbf{x}^m(\widehat{\mathbf{\theta}})$')
                plt.xlabel('t (s)')
                plt.ylabel('amplitude')
                plt.legend()
                #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 

            #gamma0_em_,gamma1_em_=autocorrelation(signal_sin-signal_sin_hat,False)[0:2]
            
            error_model=signal_sin-signal_sin_hat
            gamma0_em=np.mean(error_model**2)
            gamma1_em=np.sum(error_model[0:N-1]*error_model[1:N])/(N-1)
            
    
            #print("gamma0",gamma0,"gamma1",gamma1)
            #MSE_model[w]=(1/N)*((gamma0**2-gamma1**2)/(gamma0))
            #print((1/N)*((gamma0**2-2*gamma1**2+gamma2**2)/(gamma0)))
            #MSE_model[w]=(1/N)*((gamma0**2-2*gamma1**2+gamma2**2)/(gamma0))
            MSE_model[w]=get_mse(signal_sin, signal_sin_hat)
            
    
    
            
            theta_sin_tilde=[0]*3
            theta_sin_tilde_opt=[0]*3
            for nx in range(nx_max):
    
                
                for i in range(3):
                    theta_sin_tilde_opt[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_sin_hat[i],al_opt_int[nx,i],w_theta_sin[i],m_theta_sin[i]),al_opt_int[nx,i],w_theta_sin[i],m_theta_sin[i])
                    theta_sin_tilde[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_sin_hat[i],al_int[nx,i],w_theta_sin[i],m_theta_sin[i]),al_int[nx,i],w_theta_sin[i],m_theta_sin[i])
                #print("theta      =[{:.2f},{:.2f},{:.2f}]".format(*theta_sin))
                #print("theta tilde=[{:.2f},{:.2f},{:.2f}]".format(*theta_sin_tilde))
                signal_sin_tilde_opt=model_sin.get_model_sin(t, *theta_sin_tilde_opt)
                signal_sin_tilde=model_sin.get_model_sin(t, *theta_sin_tilde)
                
                MSE_reel_q_opt[w,nx]=get_mse(signal_sin_hat, signal_sin_tilde_opt)
                MSE_reel_q[w,nx]=get_mse(signal_sin_hat, signal_sin_tilde)
    
                MSE_reel_qm_opt[w,nx]=get_mse(signal_sin, signal_sin_tilde_opt)
                MSE_reel_qm[w,nx]=get_mse(signal_sin, signal_sin_tilde)
    
    
                
                residual_sin_opt=signal_sin-signal_sin_tilde_opt
                residual_sin=signal_sin-signal_sin_tilde
                
                residual_n_sin_opt,kr_opt=normalize(residual_sin_opt)
                residual_n_sin,kr=normalize(residual_sin)
                
                residual_n_sin_tilde_opt,code_opt=l.get_r_DCT_BPC_tilde(residual_n_sin_opt, "RMSE",-np.infty,ntot*ktot-nx+1*11)
     
                #print("code_opt",len(code_opt),ntot-nx+11)
                residual_n_sin_tilde,code=l.get_r_DCT_BPC_tilde(residual_n_sin,"RMSE",-np.infty,ntot*ktot-nx+1*11)
                
                
                

               
                
               
                #"""
                residual_n_sin_tilde_opt_2,code_opt_2=l.get_r_DWT_BPC_tilde(residual_n_sin_opt, "RMSE",-np.infty,ntot*ktot-nx+11)
                residual_n_sin_tilde_2,code_2=l.get_r_DWT_BPC_tilde(residual_n_sin,"RMSE",-np.infty,ntot*ktot-nx+11)
                if get_mse(residual_sin_opt,residual_n_sin_tilde_opt_2*2**kr_opt)<get_mse(residual_sin_opt,residual_n_sin_tilde_opt*2**kr_opt):
                 
                    residual_n_sin_tilde_opt=residual_n_sin_tilde_opt_2
                if get_mse(residual_sin,residual_n_sin_tilde_2*2**kr)<get_mse(residual_sin,residual_n_sin_tilde*2**kr):
                    residual_n_sin_tilde=residual_n_sin_tilde_2            
                #"""
                MSE_reel_r_opt[w,nx]=get_mse(residual_sin_opt,residual_n_sin_tilde_opt*2**kr_opt)
                MSE_reel_r[w,nx]=get_mse(residual_sin,residual_n_sin_tilde*2**kr)
                
                
                    
                #print("code",len(code),ntot-nx+11)
                signal_sin_rec_opt=signal_sin_tilde_opt+residual_n_sin_tilde_opt*2**kr_opt
                signal_sin_rec=signal_sin_tilde+residual_n_sin_tilde*2**kr
                
                
                
                MSE_reel_tot_opt[w,nx]=get_mse(signal_sin, signal_sin_rec_opt)
                MSE_reel_tot[w,nx]=get_mse(signal_sin, signal_sin_rec)
                
                
                error_q_opt=signal_sin_hat-signal_sin_tilde_opt
                gamma0_eq_opt=np.mean(error_q_opt**2)
                gamma1_eq_opt=np.sum(error_q_opt[0:N-1]*error_q_opt[1:N])/(N-1)
                
                error_q=signal_sin_hat-signal_sin_tilde
                gamma0_eq=np.mean(error_q**2)
                gamma1_eq=np.sum(error_q[0:N-1]*error_q[1:N])/(N-1)
                
                #gamma0_eq_opt,gamma1_eq_opt=autocorrelation(signal_sin_hat-signal_sin_tilde_opt,False)[0:2]
                #gamma0_eq,gamma1_eq=autocorrelation(signal_sin_hat-signal_sin_tilde,False)[0:2]
                            
              
                gamma0_reel_opt[w,nx]=gamma0_eq_opt
                gamma0_reel[w,nx]=gamma0_eq
                
                gamma1_reel_opt[w,nx]=gamma1_eq_opt
                gamma1_reel[w,nx]=gamma1_eq
             
                #print(al_opt_float[nx],[*al_opt_float[nx]]+[(ntot-np.sum(al_opt_float[nx]))/N])
               
                
                #MSE_model_tot_opt[w,nx]=allocation_sin.MSE_theoritical_sin([*al_opt_int[nx]]+[(ntot-np.sum(al_opt_int[nx]))/N],m_theta_sin,w_theta_sin,gamma0_em,gamma1_em)
                #MSE_model_tot[w,nx]=allocation_sin.MSE_theoritical_sin([*al_int[nx]]+[(ntot-np.sum(al_int[nx]))/N],m_theta_sin,w_theta_sin,gamma0_em,gamma1_em)
                
                MSE_model_tot_opt[w,nx]=allocation_sin.MSE_theoritical_sin([*al_opt_float[nx]]+[(ntot-nx)/N],m_theta_sin,w_theta_sin,gamma0_em,gamma1_em)
                MSE_model_tot[w,nx]=allocation_sin.MSE_theoritical_sin([*al_float[nx]]+[(ntot-nx)/N],m_theta_sin,w_theta_sin,gamma0_em,gamma1_em)
                
                
             

                    
          
                if w <1 and nx in [10,20,30]:#,20]:
                    plt.figure(figsize=(8,4), dpi=100)
                    #plt.plot(t,signal_sin_hat,lw=2,label='x hat$')
                    plt.plot(t,signal_sin_hat-signal_sin_tilde_opt,lw=2,label="error ")
                    plt.xlabel('t (s)')
                    plt.ylabel('Amplitude error q')
                    plt.legend()
                    plt.title("Erreur de quantification sinusoïdal nx={}".format(nx))
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    
                    #autocorrelation(signal_sin_hat-signal_sin_tilde_opt,True)
                    
                    
            
                    plt.figure(figsize=(8,4), dpi=100)
                    if not real:
                        plt.plot(t,signal_noise,lw=2,label=r'$\mathbf{{b}}, MSE={:.1f} µV$'.format(10**6*get_mse(signal_noise,signal_noise*0)))
                    plt.plot(t,signal_sin-signal_sin_hat,lw=2,label=r'$\mathbf{{x}}-\mathbf{{x}}^{{m}}\left(\widehat{{\mathbf{{\theta}}}}\right), MSE={:.1f} µV$'.format(10**6*get_mse(signal_sin,signal_sin_hat)))
                    plt.ylabel('amplitude')
                    plt.xlabel('t (s)')
                    plt.legend()
                    plt.title("Erreur de modélisation")
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    
                    autocorrelation(signal_sin-signal_sin_hat,True)
                    
                    

                    print("theta sin      ",[np.round(1000*theta_sin[i])/1000 for i in range (3)])
                    print("theta sin hat  ",[np.round(1000*theta_sin_hat[i])/1000 for i in range (3)])
                    print("theta sin tilde",[np.round(1000*theta_sin_tilde_opt[i])/1000 for i in range (3)])
                    
                    print("al sin float opt=[{:.2f},{:.2f},{:.2f}], nx={:.2f}, nx target={}".format(*al_opt_float[nx],np.sum(al_opt_float[nx]),nx))
                    print("al sin int opt=[{:.0f},{:.0f},{:.0f}]".format(*al_opt_int[nx]))
                    
             
            if w<10:
                root,MSE_model_tot_min = allocation_sin.get_nx_nr_constraint_bit_sin(m_theta_sin,w_theta_sin,signal_sin-signal_sin_hat,ntot,dtype)   
    
                plt.figure(figsize=(8,4), dpi=100)   
                plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot[w]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
                plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot[w]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
                plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot_opt[w]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
                plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot_opt[w]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, opt},...,n_{K, opt}\right)\right)$')
                plt.plot(np.sum(root[0:3]),10*np.log10(MSE_model_tot_min),'o',lw=4,label="Theoritical point fine")
                plt.xlabel('nx (bits)')
                plt.ylabel('10 log10 MSE (dB)')
                plt.legend()
                plt.title("Visualisation du log MSE de l'erreur de quantification sur des signaux sinusoïdaux, fenêtre={}".format(w))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()  
                
                ###########################   Voir si le débit est bien prédi
                MSE_target=MSE_model_tot_min
                print("MSE target={} dB",10*np.log10(MSE_target))
                nx_nr_find,MSE_model_tot_min_find=allocation_sin.get_nx_nr_constraint_MSE_sin(m_theta_sin,w_theta_sin,signal_sin-signal_sin_hat,MSE_target,dtype)
                print("MSE find={} dB",10*np.log10(MSE_model_tot_min_find))
                
                print("nx find ={} b, nr find={} b".format(nx_nr_find[0:len(w_theta_sin)],N*nx_nr_find[-1]),"nx taget={} b, nr target={} b".format(root[0:3],N*root[-1]))
                
    
               
                
                
                            
    
                
    
            
        ###################################################### impact de MSE
        
        plt.figure(figsize=(8,4), dpi=100)   
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_q+np.mean(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_qm,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_q_opt+np.median(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_qm_opt,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, opt},...,n_{K, opt}\right)\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("Visualisation du log MSE moyen de l'erreur de quantification sur des signaux sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        min_=np.min(np.abs(10*np.log10(MSE_model_q)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0))))
        nx=np.where(np.abs(10*np.log10(MSE_model_q)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)))==min_)[0][0]
        
        min_opt=np.min(np.abs(10*np.log10(MSE_model_q_opt)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0))))
        nx_opt=np.where(np.abs(10*np.log10(MSE_model_q_opt)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)))==min_opt)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_q,axis=0)),'*',lw=2,label=r'$\sigma_{q reel}^{2}\left((n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_q),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_q_opt,axis=0)),'*',lw=2,label=r'$\sigma_{q reel}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_q_opt),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("nx opt={}, nx={}".format(nx_opt,nx))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
            
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_r_opt,axis=0)),lw=2,label=r'$\sigma_{r,opt}^{2}$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_r,axis=0)),lw=2,label=r'$\sigma_{r}^{2}$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("nx opt={}, nx={}".format(nx_opt,nx))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        
        
        
        
        plt.figure(figsize=(8,4), dpi=100) 
        #plt.plot([i for i in range(nx_max)],np.log(MSE_model_q_opt),lw=2,label=r'MSE model')
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt),lw=2,label=r'gamma0 model')
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt),lw=2,label=r'gamma1 model')
        #plt.plot([i for i in range(nx_max)],np.log(np.mean(MSE_reel_q_opt,axis=0)),'-*',lw=2,label=r'MSE reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0)),'-*',lw=2,label=r'gamma0 reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma1_reel_opt,axis=0)),'-*',lw=2,label=r'gamma1 reel')
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()     
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(np.mean(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.mean(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt)-10*np.log10(np.mean(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel(' gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(np.median(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.median(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt)-10*np.log10(np.median(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel(' gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
          
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(gamma1_model_opt),"-*",lw=2,label=r'gamma0 model / gamma1 model, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(gamma1_model_opt))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma0_reel_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)),"-o",lw=2,label=r'gamma0 reel mean / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(np.mean(gamma0_reel_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 reel modian / gamma1 reel modian, mean={:.2f}'.format(np.mean(10*np.log10(np.median(gamma0_reel_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma0 / gamma1 (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
             
              
    
        ################################################################# mean MSE tot
    
        MSE_model_tot_opt_median=np.median(MSE_model_tot_opt,axis=0)
        min_opt_model_median=np.min(MSE_model_tot_opt_median)
        nx_opt_model_median=np.where(MSE_model_tot_opt_median == min_opt_model_median)[0][0]
        
        MSE_model_tot_opt_mean=np.mean(MSE_model_tot_opt,axis=0)
        min_opt_model_mean=np.min(MSE_model_tot_opt_mean)
        nx_opt_model_mean=np.where(MSE_model_tot_opt_mean == min_opt_model_mean)[0][0]
        
        
        
    
        MSE_model_tot_median=np.median(MSE_model_tot,axis=0)
        min_model_median=np.min(MSE_model_tot_median)
        nx_model_median=np.where(MSE_model_tot_median == min_model_median)[0][0]
    
        MSE_model_tot_mean=np.mean(MSE_model_tot,axis=0)
        min_model_mean=np.min(MSE_model_tot_mean)
        nx_model_mean=np.where(MSE_model_tot_mean == min_model_mean)[0][0]
        
            
        
        
    
        MSE_reel_tot_opt_median=np.median(MSE_reel_tot_opt,axis=0)
        min_opt_reel_median=np.min(MSE_reel_tot_opt_median)
        nx_opt_reel_median=np.where(MSE_reel_tot_opt_median == min_opt_reel_median)[0][0]
        
        MSE_reel_tot_opt_mean=np.mean(MSE_reel_tot_opt,axis=0)
        min_opt_reel_mean=np.min(MSE_reel_tot_opt_mean)
        nx_opt_reel_mean=np.where(MSE_reel_tot_opt_mean == min_opt_reel_mean)[0][0]
        
        
        
    
        MSE_reel_tot_median=np.median(MSE_reel_tot,axis=0)
        min_reel_median=np.min(MSE_reel_tot_median)
        nx_reel_median=np.where(MSE_reel_tot_median == min_reel_median)[0][0]
        
        MSE_reel_tot_mean=np.mean(MSE_reel_tot,axis=0)
        min_reel_mean=np.min(MSE_reel_tot_mean)
        nx_reel_mean=np.where(MSE_reel_tot_mean == min_reel_mean)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot_mean),lw=2,label='Mean MSE model uniform')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot_mean),'-*',lw=2,label='Mean MSE real uniform')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot_opt_mean),lw=2,label='Mean MSE model adapt')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot_opt_mean),'-*',lw=2,label='Mean MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("min m opt={:.1f}, min r opt={:.1f}, min m={:.1f}, min r={:.1f}, nx m opt={}, nx r opt={}, nx m={}, nx r={}".format(10*np.log10(min_opt_model_mean),10*np.log10(min_opt_reel_mean),10*np.log10(min_model_mean),10*np.log10(min_reel_mean),nx_opt_model_mean,nx_opt_reel_mean,nx_model_mean,nx_reel_mean))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot_median),lw=2,label='Median MSE model uniform')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot_median),'-*',lw=2,label='Median MSE real uniform')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_model_tot_opt_median),lw=2,label='Median MSE model adapt')
        plt.plot([i for i in range(nx_max)],10*np.log10(MSE_reel_tot_opt_median),'-*',lw=2,label='Median MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("min m opt={:.1f} , min r opt={:.1f}, min m={:.1f}, min r={:.1f}, nx m opt={}, nx r opt={}, nx m={}, nx r={}".format(10*np.log10(min_opt_model_median),10*np.log10(min_opt_reel_median),10*np.log10(min_model_median),10*np.log10(min_reel_median),nx_opt_model_median,nx_opt_reel_median,nx_model_median,nx_reel_median))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        #"""
        print("signal reel w={}".format(fen))
        curve_tex(t,v1[fen*N:fen*N+N]/1000,0)    
        
        print("signal reel w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_sin, signal_sin*0)/MSE_reel_tot_opt_mean),0)    
        print("signal model w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_sin, signal_sin*0)/MSE_model_tot_opt_mean),0)    
        #"""     
    









    ######################################################################  test polynome
    min_=0#4*(order+1)
    if poly :
    
       
        m_theta_poly=np.zeros(order+1)
        a=2
        b=2#00001
        order_max=15
        w_theta_poly_1=[-(a-b)*k/(order_max)+a for k in range(order_max+1)]
        tau=-np.log(b/a)/order_max
        w_theta_poly_2=[a*np.exp(-tau*k) for k in range(order_max+1)]
        #w_theta_poly=[2,1.75,1.5,1.25,1,0.75,0.5,0.25,0.25,0.1,0.1,0.05,0.05,0.01,0.01,0.01]# [2]*(order+1)
        w_theta_poly=w_theta_poly_2[0:order+1]
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(w_theta_poly_1,lw=2,label='w linéaire')
        plt.plot(w_theta_poly_2,lw=2,label='w poly')
        plt.xlabel('t (s)')
        plt.ylabel('amplitude')
        plt.legend()
        plt.title("Error model poly order {}".format(order))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
      

        
        al_opt_float=np.zeros((nx_max,order+1))
        al_opt_int=np.zeros((nx_max,order+1))
        
        al_float=np.zeros((nx_max,order+1))
        al_int=np.zeros((nx_max,order+1))
    
    
    
        for nx in range(nx_max):
           

            al_opt_float[nx]=allocation_poly.get_nx_poly(nx, w_theta_poly, dtype="float")
            al_opt_int[nx]=allocation_poly.round_allocation(al_opt_float[nx], nx)
            #print("opt float", al_opt_float[nx],np.sum(al_opt_float[nx]),nx)
            #print("opt int  ", al_opt_int[nx],np.sum(al_opt_int[nx]),nx)                    
            
                            
            MSE_model_q_opt[nx]=allocation_poly.gamma0_poly(al_opt_float[nx],w_theta_poly)
    
            gamma0_model_opt[nx]=MSE_model_q_opt[nx]
            
            
            gamma1_model_opt[nx]=allocation_poly.gamma1_poly(al_opt_float[nx],w_theta_poly)
            
        
            
            al_float[nx]=np.array([nx/(order+1)]*(order+1))
            
            al_int[nx]=allocation_poly.round_allocation(al_float[nx], nx)
            #print(" float", al_float[nx],np.sum(al_float[nx]),nx)
            #print(" int  ", al_int[nx],np.sum(al_int[nx]),nx)  

            MSE_model_q[nx]=allocation_poly.gamma0_poly(al_float[nx],w_theta_poly)
            
            gamma0_model[nx]=MSE_model_q[nx]
            gamma1_model[nx]=allocation_poly.gamma1_poly(al_float[nx],w_theta_poly)    
                
        
        for w in range(nb_signal):
            if not real :

                theta_poly=[0]*(order+1)
                for i in range(order+1):
                    theta_poly[i]=np.random.uniform(m_theta_poly[i]-w_theta_poly[i]/2,m_theta_poly[i]+w_theta_poly[i]/2)
                
                
                signal_noise=generate_corelated_noise(phi_arma,theta_arma,N,sigma,verbose)[0]
                signal_poly_clean=model_poly.get_model_poly(t, *theta_poly)
                signal_poly=signal_poly_clean+signal_noise
                k=0
            
                
                
            else: 
                signal_poly,k=normalize(v1[fen*N:fen*N+N])

            
                
            
            theta_poly_hat=model_poly.get_theta_poly(signal_poly,m_theta_poly,w_theta_poly,order)
            signal_poly_hat=model_poly.get_model_poly(t, *theta_poly_hat)
            
            error_model=signal_poly-signal_poly_hat
            
            print("SNR=",get_snr(signal_poly,signal_poly_hat))
            gamma0_em=np.mean(error_model**2)
            gamma1_em=np.sum(error_model[0:N-1]*error_model[1:N])/(N-1)
            

            MSE_model[w]=get_mse(signal_poly, signal_poly_hat)
            
           
            theta_poly_tilde=[0]*(order+1)
            theta_poly_tilde_opt=[0]*(order+1)
            
            
            test_error_model=1
            if test_error_model:
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,signal_poly,lw=2,label=r'$\mathbf{x}$')
                plt.plot(t,signal_poly_hat,lw=2,label=r'$\mathbf{x}^m(\widehat{\mathbf{\theta}})$')
                plt.xlabel('t (s)')
                plt.ylabel('amplitude')
                plt.legend()
                plt.title("poly-{}".format(order))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
                
                plt.figure(figsize=(8,4), dpi=100)
                if not real :
                    plt.plot(t,signal_noise,lw=2,label=r'$\mathbf{{b}}, MSE={:.1f} dB$'.format(10*np.log10(get_mse(signal_noise,signal_noise*0))))
                plt.plot(t,signal_poly-signal_poly_hat,lw=2,label=r'$\mathbf{{x}}-\mathbf{{x}}^{{m}}\left(\widehat{{\mathbf{{\theta}}}}\right), MSE={:.8f}$'.format(10*np.log10(get_mse(signal_poly,signal_poly_hat))))
                plt.ylabel('Amplitude (V)')
                plt.xlabel('t (s)')
                plt.legend()
                plt.title("Modeling error poly-{}".format(order))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
                
                autocorrelation(signal_poly-signal_poly_hat,True)
        
        
            test_AR=0
            if test_AR:
              
                # instantiate model objet
                model=ARIMA(error_model,order=(2,0,0))
                results=model.fit()
                print(results.summary())
                
                # Simulate data starting at the end of the time series
                rep=3
                #np.random.seed(8659567)  # Remplacez 123 par la valeur de votre choix
                sim = results.simulate(N,repetitions=rep)
         
    
                            
                plt.figure(figsize=(8,4), dpi=100)
                #plt.plot(t,signal_sin_hat,lw=2,label='x hat$')
                plt.plot(t,signal_poly-signal_poly_hat,lw=2,label="error")
                for k in range(rep):
                    plt.plot(t,sim[:,0,k],lw=2,label="error generate")
                plt.xlabel('t (s)')
                plt.ylabel('Amplitude error q')
                plt.legend()
                plt.title("erreur model")
                plt.title("Visualisation du log de MSE  due à la quantification poly order {}".format(order))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
    
      
          
            for nx in range(0,nx_max):
    
                
                for i in range(order+1):
                    theta_poly_tilde_opt[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_poly_hat[i],al_opt_int[nx,i],w_theta_poly[i],m_theta_poly[i]),al_opt_int[nx,i],w_theta_poly[i],m_theta_poly[i])
                    theta_poly_tilde[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_poly_hat[i],al_int[nx,i],w_theta_poly[i],m_theta_poly[i]),al_int[nx,i],w_theta_poly[i],m_theta_poly[i])
                
    
                    
    
                signal_poly_tilde_opt=model_poly.get_model_poly(t, *theta_poly_tilde_opt)
                signal_poly_tilde=model_poly.get_model_poly(t, *theta_poly_tilde)
                
                MSE_reel_q_opt[w,nx]=get_mse(signal_poly_hat, signal_poly_tilde_opt)
                MSE_reel_q[w,nx]=get_mse(signal_poly_hat, signal_poly_tilde)
                
                
                MSE_reel_qm_opt[w,nx]=get_mse(signal_poly, signal_poly_tilde_opt)
                MSE_reel_qm[w,nx]=get_mse(signal_poly, signal_poly_tilde)
    
    
                
                residual_poly_opt=signal_poly-signal_poly_tilde_opt
                residual_poly=signal_poly-signal_poly_tilde
                
                _,kr=normalize(residual_poly)
     
                _,kr_opt=normalize(residual_poly_opt)
                kr=0
                kr_opt=0
            
                
                residual_n_poly=np.copy(residual_poly)*2**(-kr)
                residual_n_poly_opt=np.copy(residual_poly_opt)*2**(-kr_opt)
                
                
                
                
                #print(kr,kr_opt)
                residual_n_poly_tilde_opt,code_opt=l.get_r_DCT_BPC_tilde(residual_n_poly_opt, "RMSE",-np.infty,ntot*ktot-nx+11)
                #print(ntot-nx+11)
                #print("code_opt",len(code_opt),ntot-nx+11)
                residual_n_poly_tilde,code=l.get_r_DCT_BPC_tilde(residual_n_poly,"RMSE",-np.infty,ntot*ktot-nx+11)
    
                #print("code",len(code),ntot-nx+11)
                #"""
                residual_n_poly_tilde_opt_2,code_opt_2=l.get_r_DWT_BPC_tilde(residual_n_poly_opt, "RMSE",-np.infty,ntot*ktot-nx+11)
                residual_n_poly_tilde_2,code_2=l.get_r_DWT_BPC_tilde(residual_n_poly,"RMSE",-np.infty,ntot*ktot-nx+11)
                if get_mse(residual_poly_opt,residual_n_poly_tilde_opt_2*2**kr_opt)<get_mse(residual_poly_opt,residual_n_poly_tilde_opt*2**kr_opt):
                 
                    residual_n_poly_tilde_opt=np.copy(residual_n_poly_tilde_opt_2)
                if get_mse(residual_poly,residual_n_poly_tilde_2*2**kr)<get_mse(residual_poly,residual_n_poly_tilde*2**kr):
                    residual_n_poly_tilde=np.copy(residual_n_poly_tilde_2   )         
                #"""
                
                
                
                signal_poly_rec_opt=signal_poly_tilde_opt+residual_n_poly_tilde_opt*2**(kr_opt)
                signal_poly_rec=signal_poly_tilde+residual_n_poly_tilde*2**(kr)
                
                
                
                MSE_reel_r_opt[w,nx]=get_mse(residual_poly_opt,residual_n_poly_tilde_opt*2**(kr_opt))
                MSE_reel_r[w,nx]=get_mse(residual_poly,residual_n_poly_tilde*2**(kr))
                
                
                MSE_reel_tot_opt[w,nx]=get_mse(signal_poly, signal_poly_rec_opt)
                MSE_reel_tot[w,nx]=get_mse(signal_poly, signal_poly_rec)
                
    
    
                error_q_opt=signal_poly_hat-signal_poly_tilde_opt
                
                
                scalar_product_m[w,nx]=np.mean(error_model**2)
                scalar_product_mq[w,nx]=np.mean(error_q_opt*error_model)
                scalar_product_q[w,nx]=np.mean(error_q_opt**2)
                
                gamma0_eq_opt=np.mean(error_q_opt**2)
                gamma1_eq_opt=np.sum(error_q_opt[0:N-1]*error_q_opt[1:N])/(N-1)
                
                error_q=signal_poly_hat-signal_poly_tilde
                gamma0_eq=np.mean(error_q**2)
                gamma1_eq=np.sum(error_q[0:N-1]*error_q[1:N])/(N-1)
                #gamma0_eq_opt,gamma1_eq_opt=autocorrelation(signal_poly_hat-signal_poly_tilde_opt,False)[0:2]
                #gamma0_eq,gamma1_eq=autocorrelation(signal_poly_hat-signal_poly_tilde,False)[0:2]
                            
                
                gamma0_reel_opt[w,nx]=gamma0_eq_opt
                gamma0_reel[w,nx]=gamma0_eq
                
                gamma1_reel_opt[w,nx]=gamma1_eq_opt
                gamma1_reel[w,nx]=gamma1_eq
                
                
                MSE_model_tot_opt[w,nx]=allocation_poly.MSE_theoritical_poly([*al_opt_float[nx]]+[(ntot-nx)/N],w_theta_poly,gamma0_em,gamma1_em)
                MSE_model_tot[w,nx]=allocation_poly.MSE_theoritical_poly([*al_float[nx]]+[(ntot-nx)/N],w_theta_poly,gamma0_em,gamma1_em)
                
                #MSE_model_tot_opt[w,nx]=allocation_poly.MSE_theoritical_poly([*al_opt_int[nx]]+[(ntot-np.sum(al_opt_int[nx]))/N],w_theta_poly,gamma0_em,gamma1_em)
                #MSE_model_tot[w,nx]=allocation_poly.MSE_theoritical_poly([*al_int[nx]]+[(ntot-np.sum(al_int[nx]))/N],w_theta_poly,gamma0_em,gamma1_em)
                
                #print(np.sum(al_opt_float[nx]),np.sum(al_opt_int[nx]))
                

                    
                    
            
        
                if w <0 and nx in [0,19,20,21,22,23]:
                    plt.figure(figsize=(8,4), dpi=100)
                    plt.plot(t,signal_poly_hat,lw=2,label='x hat')
                    plt.plot(t,signal_poly_tilde_opt,lw=2,label="x tilde adapted, MSE={:.3f} dB ".format(10*np.log10(get_mse(signal_poly_hat,signal_poly_tilde_opt))))
                    plt.plot(t,signal_poly_tilde,lw=2,label="x tilde uniform, MSE={:.3f} dB ".format(10*np.log10(get_mse(signal_poly_hat,signal_poly_tilde))))
                    plt.xlabel('t (s)')
                    plt.ylabel('Amplitude (V)')
                    plt.legend()
                    plt.title("Reconstructed model poly-{} , nx={} bits".format(order,nx))
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    

                    
                    plt.figure(figsize=(8,4), dpi=100)
                    plt.plot(t,signal_poly_hat-signal_poly_tilde_opt,lw=2,label="x hat  - x tilde adapted, MSE= {:.3f} ".format(10*np.log10(get_mse(signal_poly_hat,signal_poly_tilde_opt))))
                    plt.plot(t,signal_poly_hat-signal_poly_tilde,lw=2,label="x hat - x tilde uniform,  MSE= {:.3f} ".format(10*np.log10(get_mse(signal_poly_hat,signal_poly_tilde))))
                    plt.xlabel('t (s)')
                    plt.ylabel('Amplitude (V)')
                    plt.legend()
                    plt.title("Error reconstructed model poly-{}, nx={}".format(order,nx))
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    
                    #print("max residual opt",np.max(np.abs(residual_n_poly_opt)))
                    plt.figure(figsize=(8,4), dpi=100)
                    plt.plot(t,residual_n_poly_opt,lw=2,label='r adapted, nx={} b'.format(nx))
                    plt.plot(t,residual_n_poly_tilde_opt,lw=2,label="r tilde adapted, MSE={:.3f} dB".format(10*np.log10(get_mse(residual_poly_opt,residual_n_poly_tilde_opt*2**(kr_opt)))))
                    plt.plot(t,residual_n_poly,lw=2,label='r uniform, nx={} b'.format(nx))
                    plt.plot(t,residual_n_poly_tilde,lw=2,label="r tilde uniform, , MSE={:.3f} dB".format(10*np.log10(get_mse(residual_poly,residual_n_poly_tilde*2**(kr)))))
                    plt.xlabel('t (s)')
                    plt.ylabel('Amplitude (V)')
                    plt.legend()
                    plt.title("Error reconstructed residual poly-{}, nr={} bits".format(order,ntot-nx))
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
         
                    #autocorrelation(signal_poly_hat-signal_poly_tilde_opt,True)
                    
                    print("-----------------------------------")
                    if not real :
                        print("theta poly-{}      ".format(order),[np.round(1000*theta_poly[i])/1000 for i in range (order+1)])
                    print("w theta poly-{}            ".format(order),[np.round(1000*w_theta_poly[i])/1000 for i in range (order+1)])
                    print("theta poly-{} hat          ".format(order),[np.round(1000*theta_poly_hat[i])/1000 for i in range (order+1)])
                    print("theta poly-{} tilde adapted".format(order),[np.round(1000*theta_poly_tilde_opt[i])/1000 for i in range (order+1)])
                    print("theta poly-{} tilde uniform".format(order),[np.round(1000*theta_poly_tilde[i])/1000 for i in range (order+1)])
                    
                    print("Alocation poly-{} float adapted={}, nx={:.2f}, nx target={}".format(order,[np.round(10000*al_opt_float[nx,i])/10000 for i in range(order+1)],np.sum(al_opt_float[nx]),nx))
                    print("Alocation poly-{} int   adapted={}, nx={:.2f}, nx target={}".format(order,al_opt_int[nx],np.sum(al_opt_int[nx]),nx))
                    
                    print("Alocation poly-{} float uniform={}, nx={:.2f}, nx target={}".format(order,[np.round(10000*al_float[nx,i])/10000 for i in range(order+1)],np.sum(al_float[nx]),nx))
                    print("Alocation poly-{} int   uniform={}, nx={:.2f}, nx target={}".format(order,al_int[nx],np.sum(al_int[nx]),nx))
            if w<10:# in [0,10,20,30,40,50,60,70,80,100]:
                root,MSE_model_tot_min = allocation_poly.get_nx_nr_constraint_bit_poly(w_theta_poly,signal_poly-signal_poly_hat,ntot,dtype)   
            
                          
                plt.figure(figsize=(8,4), dpi=100)   
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_tot[w][min_:]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_reel_tot[w][min_:]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_tot_opt[w][min_:]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, ad},...,n_{K, ad}\right)$')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_reel_tot_opt[w][min_:]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, ad},...,n_{K, ad}\right)\right)$')
                plt.plot(np.sum(root[0:order+1]),10*np.log10(np.mean(signal_poly**2)/MSE_model_tot_min),'o',lw=4,label="Theoritical point")
                plt.xlabel('nx (bits)')
                plt.ylabel('SNR (dB)')
                plt.legend()
                plt.title("SNR of the quantization error poly-{} depending on nx".format(order))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()  
                
                ###########################   Voir si le débit est bien prédi
                MSE_target=MSE_model_tot_min
                print("SNR target={:.2f} dB".format(10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_target)))
                nx_nr_find,MSE_model_tot_min_find=allocation_poly.get_nx_nr_constraint_MSE_poly(w_theta_poly,signal_poly-signal_poly_hat,MSE_target,dtype)
                print("SNR found={:.2f} dB".format(10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_model_tot_min_find)))
                
                print("nx found={} b, nr found={} b".format(nx_nr_find,np.round(10*N*nx_nr_find[-1])/10))
                print("nx taget={} b, nr target={} b".format(root[0:len(w_theta_poly)],np.round(10*N*root[-1])/10))
                
    
                
    
                #print("al poly int opt=[{:.0f},{:.0f},{:.0f}]".format(*al_poly_opt_int))
                

        plt.figure(figsize=(8,4), dpi=100)   
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/(MSE_model_q+np.mean(MSE_model,axis=0))),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_qm,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/(MSE_model_q_opt+np.mean(MSE_model,axis=0))),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, ad},...,n_{K, ad}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_qm_opt,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, ad},...,n_{K, ad}\right)\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title("SNR of the modeling and quantization error poly-{}".format(order))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        #min_nx=np.max(np.abs(10*np.log10(np.mean(signal_poly**2)/MSE_model_q)-10*np.log10(np.mean(signal_poly**2)/(np.ones(nx_max)*np.mean(MSE_model,axis=0)))))
        
        #nx=np.where(np.abs(10*np.log10(np.mean(signal_poly**2)/MSE_model_q)-10*np.log10(np.mean(signal_poly**2)/np.ones(nx_max)*np.mean(MSE_model,axis=0)))==min_nx)[0][0]
        
        #min_opt=np.max(np.abs(10*np.log10(np.mean(signal_poly**2)/MSE_model_q_opt)-10*np.log10(np.mean(signal_poly**2)/np.ones(nx_max)*np.mean(MSE_model,axis=0))))
        #nx_opt=np.where(np.abs(10*np.log10(np.mean(signal_poly**2)/MSE_model_q_opt)-10*np.log10(np.mean(signal_poly**2)/np.ones(nx_max)*np.mean(MSE_model,axis=0)))==min_opt)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_q,axis=0)),'*',lw=2,label=r'$\sigma_{q real}^{2}\left((n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_q),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_q_opt,axis=0)),'*',lw=2,label=r'$\sigma_{q real}^{2}\left(n_{1, ad},...,n_{K, ad}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_q_opt),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1, ad},...,n_{K, ad}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(signal_poly**2)/(np.ones(nx_max)*np.mean(MSE_model,axis=0))),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title("SNR depending of nx for error estimation and quantization")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        

        
        SNR_em_eq_model=10*np.log10(np.mean(signal_poly**2)/(MSE_model_q_opt+np.mean(MSE_model,axis=0)))
        SNR_em_eq_real=10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_qm,axis=0))
        
        SNR_em_eq_res_real=10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_reel_r_opt,axis=0))
        SNR_em_eq_res_model=10*np.log10(np.mean(signal_poly**2)/np.mean(MSE_model_tot_opt,axis=0))
        
        SNR_est=10*np.log10(np.mean(signal_poly**2)/(np.ones(nx_max)*np.mean(MSE_model,axis=0)))
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],SNR_em_eq_model,lw=2,label="est+q model")
        plt.plot([i for i in range(nx_max)],SNR_em_eq_real,'*',lw=2,label="est+q real")
        #plt.plot([i for i in range(nx_max)],SNR_est,lw=2,label="estimaration")
        #plt.plot([i for i in range(nx_max)],SNR_em_eq_res_real,'*',lw=2,label='model+res')
        plt.plot([i for i in range(nx_max)],SNR_em_eq_res_model-SNR_em_eq_model,lw=2,label='model+res')
        plt.plot([i for i in range(nx_max)],SNR_em_eq_res_real-SNR_em_eq_real,'*',lw=2,label='model+res')
        plt.xlabel('nx (bits)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title("Contribution of the first and second stage")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        print("SNR_em_eq_model w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],SNR_em_eq_model,0)   
        print("SNR_em_eq_real w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],SNR_em_eq_real,0) 
        print("SNR_em_eq_res_model w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],SNR_em_eq_res_model-SNR_em_eq_model,0) 
        print("SNR_em_eq_res_real w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],SNR_em_eq_res_real-SNR_em_eq_real,0) 
        
   
    
        
        
        
        
        plt.figure(figsize=(8,4), dpi=100) 
        #plt.plot([i for i in range(nx_max)],np.log(MSE_model_q_opt),lw=2,label=r'MSE model')
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt),lw=2,label=r'gamma0 model')
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt),lw=2,label=r'gamma1 model')
        #plt.plot([i for i in range(nx_max)],np.log(np.mean(MSE_reel_q_opt,axis=0)),'-*',lw=2,label=r'MSE reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0)),'-*',lw=2,label=r'gamma0 reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma1_reel_opt,axis=0)),'-*',lw=2,label=r'gamma1 reel')
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()     
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(np.mean(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.mean(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt)-10*np.log10(np.mean(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(scalar_product_m,axis=0)+np.mean(scalar_product_q,axis=0)),lw=2,label=r'10\log<error modeling,error modeling>+<error q,error q>')
        plt.plot([i for i in range(nx_max)],10*np.log10(2*np.abs(np.mean(scalar_product_mq,axis=0))),lw=2,label=r'10\log2|<error modeling,error q>|')
        plt.xlabel('nx (bits)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.title("Verification of the bit allocation model hypothesis")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        
        """
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(np.median(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.median(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma1_model_opt)-10*np.log10(np.median(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        """  
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(gamma0_model_opt)-10*np.log10(gamma1_model_opt),"-*",lw=2,label=r'gamma0 model / gamma1 model, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(gamma1_model_opt))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma0_reel_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)),"-o",lw=2,label=r'gamma0 reel mean / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(np.mean(gamma0_reel_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        #plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 reel modian / gamma1 reel modian, mean={:.2f}'.format(np.mean(10*np.log10(np.median(gamma0_reel_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma0 / gamma1 (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
         
              
        """
        ################################################################# mean MSE tot
    
        MSE_model_tot_opt_median=np.median(MSE_model_tot_opt,axis=0)
        min_opt_model_median=np.min(MSE_model_tot_opt_median)
        nx_opt_model_median=np.where(MSE_model_tot_opt_median == min_opt_model_median)[0][0]
        
        MSE_model_tot_opt_mean=np.mean(MSE_model_tot_opt,axis=0)
        min_opt_model_mean=np.min(MSE_model_tot_opt_mean)
        nx_opt_model_mean=np.where(MSE_model_tot_opt_mean == min_opt_model_mean)[0][0]
        
        
        
    
        MSE_model_tot_median=np.median(MSE_model_tot,axis=0)
        min_model_median=np.min(MSE_model_tot_median)
        nx_model_median=np.where(MSE_model_tot_median == min_model_median)[0][0]
    
        MSE_model_tot_mean=np.mean(MSE_model_tot,axis=0)
        min_model_mean=np.min(MSE_model_tot_mean)
        nx_model_mean=np.where(MSE_model_tot_mean == min_model_mean)[0][0]
        
            
        
        
    
        MSE_reel_tot_opt_median=np.median(MSE_reel_tot_opt,axis=0)
        min_opt_reel_median=np.min(MSE_reel_tot_opt_median)
        nx_opt_reel_median=np.where(MSE_reel_tot_opt_median == min_opt_reel_median)[0][0]
        
        MSE_reel_tot_opt_mean=np.mean(MSE_reel_tot_opt,axis=0)
        min_opt_reel_mean=np.min(MSE_reel_tot_opt_mean)
        nx_opt_reel_mean=np.where(MSE_reel_tot_opt_mean == min_opt_reel_mean)[0][0]
        
        
        
    
        MSE_reel_tot_median=np.median(MSE_reel_tot,axis=0)
        min_reel_median=np.min(MSE_reel_tot_median)
        nx_reel_median=np.where(MSE_reel_tot_median == min_reel_median)[0][0]
        
        MSE_reel_tot_mean=np.mean(MSE_reel_tot,axis=0)
        min_reel_mean=np.min(MSE_reel_tot_mean)
        nx_reel_mean=np.where(MSE_reel_tot_mean == min_reel_mean)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_tot_mean)[min_:],lw=2,label='MSE model uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_reel_tot_mean)[min_:],'-*',lw=2,label='MSE real uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_model_tot_opt_mean)[min_:],lw=2,label='MSE model adapt')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(np.mean(signal_poly**2)/MSE_reel_tot_opt_mean)[min_:],'-*',lw=2,label='MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title("min model={:.1f}, min real opt={:.1f}, nx model={}, nx real={}".format(10*np.log10(min_opt_model_mean),10*np.log10(min_opt_reel_mean),nx_opt_model_mean,nx_opt_reel_mean))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        """
        """
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_model_tot_median[min_:]),lw=2,label='Median MSE model uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_reel_tot_median[min_:]),'-*',lw=2,label='Median MSE real uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_model_tot_opt_median[min_:]),lw=2,label='Median MSE model adapt')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(2**(-2*k)*np.mean(signal_poly**2)/MSE_reel_tot_opt_median[min_:]),'-*',lw=2,label='Median MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title("min model={:.1f}, min real opt={:.1f}, nx model={}, nx real={}".format(10*np.log10(min_opt_model_mean),10*np.log10(min_opt_reel_mean),nx_opt_model_mean,nx_opt_reel_mean))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 

        print("signal reel w={}".format(fen))
        curve_tex(t,v1[fen*N:fen*N+N]/1000,0)    
        
        print("signal reel w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_poly, signal_poly*0)/MSE_reel_tot_opt_mean),0)    
        print("signal model w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_poly, signal_poly*0)/MSE_model_tot_opt_mean),0)    
        #"""     

    ###################################################################### test pred_sample

    if pred_samples:
       
        m_theta_pred_samples=np.zeros((nb_signal,order_pred))
        if real:
            w_theta_pred_samples=[2]*order_pred
        else :
            w_theta_pred_samples=np.array([0.5,0.01,0.4,0.01,0.2,0.01,0.8,0.1,0.5])
            w_theta_pred_samples=w_theta_pred_samples[0:order_pred]
        ########################################### on test si l'allocation de bits nx est bonne
        
    
    
        MSE_model=np.zeros(nb_signal)
        
        MSE_model_q=np.zeros((nb_signal,nx_max))
        MSE_model_q_opt=np.zeros((nb_signal,nx_max))
        MSE_reel_q=np.zeros((nb_signal,nx_max))
        MSE_reel_q_opt=np.zeros((nb_signal,nx_max))
        
     
        MSE_reel_qm=np.zeros((nb_signal,nx_max))
        MSE_reel_qm_opt=np.zeros((nb_signal,nx_max))    
        
        MSE_model_tot=np.zeros((nb_signal,nx_max))
        MSE_model_tot_opt=np.zeros((nb_signal,nx_max))
        MSE_reel_tot=np.zeros((nb_signal,nx_max))
        MSE_reel_tot_opt=np.zeros((nb_signal,nx_max))
        
        
    
    
    
    
        
        m_theta_sin=[0.75,fn,0]
        w_theta_sin=[0.5,0.2,2*np.pi]
        
        tp=np.linspace(-3*N/fs,0-1/(N*fs),3*N)
        
        
        signal_pred_samples_hat=np.zeros((nb_signal,N))
        signal_pred_samples=np.zeros((nb_signal,N))
        theta_hat=np.zeros((nb_signal,order_pred))
        signal_previous=np.zeros((nb_signal,3*N))
        
        X=[]
        
        gamma0_em=np.zeros(nb_signal)
        gamma1_em=np.zeros(nb_signal)
        
        gamma0_q=np.zeros(nb_signal)
        gamma1_q=np.zeros(nb_signal)
        
        
        
        
        
        
        al_opt_float=np.zeros((nx_max,order_pred))
        al_opt_int=np.zeros((nx_max,order_pred))
        
        al_float=np.zeros((nx_max,order_pred))
        al_int=np.zeros((nx_max,order_pred))
        
        gamma0_reel=np.zeros((nb_signal,nx_max))
        gamma0_reel_opt=np.zeros((nb_signal,nx_max))
        gamma0_model_opt=np.zeros((nb_signal,nx_max))
        gamma0_model=np.zeros((nb_signal,nx_max))
        
        gamma1_reel_opt=np.zeros((nb_signal,nx_max))
        gamma1_model_opt=np.zeros((nb_signal,nx_max))
        gamma1_reel=np.zeros((nb_signal,nx_max))
        gamma1_model=np.zeros((nb_signal,nx_max))
        
        
        
        for nx in range(nx_max):
            
    
            #print(allocation_poly.get_nx_poly(10, w_theta_poly, dtype="float"))
            al_opt_float[nx]=allocation_pred_samples.get_nx_pred_samples(nx,w_theta_pred_samples,eta,dtype="float")
            al_opt_int[nx]=allocation_pred_samples.round_allocation(al_opt_float[nx], nx)
    
                
            al_float[nx]=[nx/(order_pred)]*(order_pred)
            al_int[nx]=allocation_pred_samples.round_allocation(al_float[nx], nx)
    
    
    
    
        for w in range(nb_signal):
            
            
            # génération des signaux x rec 
            a=np.random.uniform(m_theta_sin[0]-w_theta_sin[0]/2,m_theta_sin[0]+w_theta_sin[0]/2)
            f=np.random.uniform(m_theta_sin[1]-w_theta_sin[1]/2,m_theta_sin[1]+w_theta_sin[1]/2)
            phi=np.random.uniform(m_theta_sin[2]-w_theta_sin[2]/2,m_theta_sin[2]+w_theta_sin[2]/2)
            
            a2=np.random.uniform(0,0.2)
            f2=3*f
            phi2=np.random.uniform(-np.pi,np.pi)
            
            
            theta_sin=[a,f,phi]
            theta_sin2=[a2,f2,phi2]
            
           
            signal_noise=generate_corelated_noise(phi_arma,theta_arma,3*N,sigma,False)[0]
            
            
            if real:
                signal_previous[w],K=normalize(v1[(fen-3)*N:(fen-3)*N+3*N])
            else :
                signal_previous[w]=model_sin.get_model_sin(tp,*theta_sin)+model_sin.get_model_sin(tp,*theta_sin2)+signal_noise
            
            
            #print("signal_previous[w]",signal_previous[w])
            
            ############## définir m_theta
            X2=model_pred_samples.get_X(signal_previous[w][0:2*N], order_pred, eta)
            m_theta_pred_samples[w]=model_pred_samples.get_theta_pred_samples(X2,signal_previous[w][2*N:3*N],m_theta_pred_samples[w],[100]*order_pred)
    
            
            
            # génération signal via pred samples
            
            X.append(model_pred_samples.get_X(signal_previous[w][N:3*N], order_pred, eta))
    
            
            
            theta_pred_samples=[0]*(order_pred)
            #while np.abs(np.sum(theta_pred_samples)-1)>0.01:
                
            if real:
                #theta_pred_samples=model_pred_samples.get_theta_pred_samples(X[w], v1[(fen)*N:(fen)*N+N]*2**(-K), m_theta_pred_samples[w], w_theta_pred_samples)
                signal_pred_samples[w]=v1[(fen)*N:(fen)*N+N]*2**(-K)
                #model_pred_samples.get_model_pred_samples(X[w],*theta_pred_samples)    
            else :
                
                while np.abs(1-np.sum(theta_pred_samples))>0.5:
                    for i in range(order_pred):
                            theta_pred_samples[i]=np.random.uniform(m_theta_pred_samples[w][i]-w_theta_pred_samples[i]/2,m_theta_pred_samples[w][i]+w_theta_pred_samples[i]/2)
                       
          
                
                signal_noise=generate_corelated_noise(phi_arma,theta_arma,N,sigma,False)[0]
                
                signal_pred_samples[w]=model_pred_samples.get_model_pred_samples(X[w],*theta_pred_samples)+signal_noise            
                
            
        
            
            theta_hat[w]=model_pred_samples.get_theta_pred_samples(X[w],signal_pred_samples[w],m_theta_pred_samples[w],w_theta_pred_samples)
            signal_pred_samples_hat[w]=model_pred_samples.get_model_pred_samples(X[w], *theta_hat[w])
            
    
            #gamma0_em_,gamma1_em_=autocorrelation(signal_sin-signal_sin_hat,False)[0:2]
            
            error_model=signal_pred_samples[w]-signal_pred_samples_hat[w]
            #print("signal_pred_samples[w]",signal_pred_samples[w])
            #print("signal_pred_samples_hat[w]",signal_pred_samples_hat[w])
            #print("error_model",error_model)
            
            
            gamma0_em[w]=np.mean(error_model**2)
            gamma1_em[w]=np.sum(error_model[0:N-1]*error_model[1:N])/(N)
            #gamma1_em[w]=np.min([np.mean(error_model[0:N-1]*error_model[1:N]),(1-1/N)*gamma0_em[w]])
            
            #print("gamma0_em[w]",gamma0_em[w])
            #print("gamma1_em[w]",gamma1_em[w])
        
            #print("gamma0",gamma0,"gamma1",gamma1)
            #MSE_model[w]=(1/N)*((gamma0**2-gamma1**2)/(gamma0))
            #print((1/N)*((gamma0**2-2*gamma1**2+gamma2**2)/(gamma0)))
            #MSE_model[w]=(1/N)*((gamma0**2-2*gamma1**2+gamma2**2)/(gamma0))
            MSE_model[w]=gamma0_em[w]#get_mse(signal_sin, signal_sin_hat)
            
            
            
            gamma0_q[w]=np.mean(signal_previous[w][2*N:3*N]**2)
            gamma1_q[w]=np.sum(signal_previous[w][2*N:3*N-1]*signal_previous[w][2*N+1:3*N])/(N)
            
            """
            if gamma1_q[w]>gamma0_q[w]:
                print("av",gamma1_q[w],gamma0_q[w])
                gamma1_q[w]=gamma0_q[w]-(gamma1_q[w]-gamma0_q[w])
                print("ap",gamma1_q[w],gamma0_q[w])
            """
       
            #gamma1_q[w]=np.min([np.mean(signal_previous[w][2*N:3*N-1]*signal_previous[w][2*N+1:3*N]),(1-1/N)*gamma0_q[w]])
       
            
            test_model_error=0
            if test_model_error and w<3:
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(tp,signal_previous[w],lw=2,label="previous")
                plt.plot(t,signal_pred_samples[w],lw=2,label=r'$\mathbf{x}$')
                plt.plot(t,signal_pred_samples_hat[w],lw=2,label=r'$\mathbf{x}^m(\widehat{\mathbf{\theta}})$')
                plt.xlabel('t (s)')
                plt.ylabel('amplitude')
                plt.legend()
                #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
                print("m_theta_pred_samples  ",[np.round(1000*m_theta_pred_samples[w][i])/1000 for i in range (order_pred)],np.sum(m_theta_pred_samples[w]))
                print("theta_pred_samples    ",[np.round(1000*theta_pred_samples[i])/1000 for i in range (order_pred)],np.sum(theta_pred_samples))
                print("theta_pred samples hat",[np.round(1000*theta_hat[w][i])/1000 for i in range (order_pred)],np.sum(theta_hat[w]))
                
                print("al pred float opt={}, nx={:.2f}, nx target={}".format([np.round(100*al_opt_float[nx,i])/100 for i in range(order_pred)],np.sum(al_opt_float[nx]),nx))
                #print("al poly int opt=[{:.0f},{:.0f},{:.0f}]".format(*al_poly_opt_int))
                
        
            
         
            for nx in range(nx_max):
                
    
                MSE_model_q_opt[w][nx]=allocation_pred_samples.gamma0_pred_samples(al_opt_float[nx],w_theta_pred_samples,gamma0_q[w])
                 
                
                
                gamma0_model_opt[w][nx]=MSE_model_q_opt[w][nx]
                
                
                gamma1_model_opt[w][nx]=allocation_pred_samples.gamma1_pred_samples(al_opt_float[nx],w_theta_pred_samples,gamma0_q[w],gamma1_q[w])
                
    
                MSE_model_q[w][nx]=allocation_pred_samples.gamma0_pred_samples(al_float[nx],w_theta_pred_samples,gamma0_q[w])
                
                gamma0_model[w][nx]=MSE_model_q[w][nx]
                gamma1_model[w][nx]=allocation_pred_samples.gamma1_pred_samples(al_float[nx],w_theta_pred_samples,gamma0_q[w],gamma1_q[w])    
                    
    
        
        for w in range(nb_signal):
            for nx in range(nx_max):
    
                theta_pred_samples_tilde_opt=np.zeros(order_pred)
                theta_pred_samples_tilde=np.zeros(order_pred)
                for i in range(order_pred):
                    theta_pred_samples_tilde_opt[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_hat[w][i],al_opt_int[nx,i],w_theta_pred_samples[i],m_theta_pred_samples[w][i]),al_opt_int[nx,i],w_theta_pred_samples[i],m_theta_pred_samples[w][i])
                    theta_pred_samples_tilde[i]=quantizer.get_q_u(quantizer.get_ind_u(theta_hat[w][i],al_int[nx,i],w_theta_pred_samples[i],m_theta_pred_samples[w][i]),al_int[nx,i],w_theta_pred_samples[i],m_theta_pred_samples[w][i])
                
    
                #print("nx---------",nx)
                #print(theta_hat[w])
                #print(theta_pred_samples_tilde_opt)   
                signal_pred_samples_tilde_opt=model_pred_samples.get_model_pred_samples(X[w], *theta_pred_samples_tilde_opt)
                signal_pred_samples_tilde=model_pred_samples.get_model_pred_samples(X[w], *theta_pred_samples_tilde)
                
                MSE_reel_q_opt[w,nx]=get_mse(signal_pred_samples_hat[w], signal_pred_samples_tilde_opt)
                MSE_reel_q[w,nx]=get_mse(signal_pred_samples_hat[w], signal_pred_samples_tilde)
                
                
                MSE_reel_qm_opt[w,nx]=get_mse(signal_pred_samples[w], signal_pred_samples_tilde_opt)
                MSE_reel_qm[w,nx]=get_mse(signal_pred_samples[w], signal_pred_samples_tilde)
        
        
                
                residual_pred_samples_opt=signal_pred_samples[w]-signal_pred_samples_tilde_opt
                residual_pred_samples=signal_pred_samples[w]-signal_pred_samples_tilde
                
                residual_n_pred_samples_opt,kr_opt=normalize(residual_pred_samples_opt)
                residual_n_pred_samples,kr=normalize(residual_pred_samples)
                
                residual_n_pred_samples_tilde_opt,code_opt=l.get_r_DCT_BPC_tilde(residual_n_pred_samples_opt, "RMSE",-np.infty,ntot*ktot-nx+11)
                #print("code_opt",len(code_opt),ntot-nx+11)
                residual_n_pred_samples_tilde,code=l.get_r_DCT_BPC_tilde(residual_n_pred_samples,"RMSE",-np.infty,ntot*ktot-nx+11)
                
                #"""
                residual_n_pred_samples_tilde_opt_2,code_opt_2=l.get_r_DWT_BPC_tilde(residual_n_pred_samples_opt, "RMSE",-np.infty,ntot*ktot-nx+11)
                residual_n_pred_samples_tilde_2,code_2=l.get_r_DWT_BPC_tilde(residual_n_pred_samples,"RMSE",-np.infty,ntot*ktot-nx+11)
                if get_mse(residual_pred_samples_opt,residual_n_pred_samples_tilde_opt_2*2**kr_opt)<get_mse(residual_pred_samples_opt,residual_n_pred_samples_tilde_opt*2**kr_opt):
                 
                    residual_n_pred_samples_tilde_opt=np.copy(residual_n_pred_samples_tilde_opt_2)
                if get_mse(residual_pred_samples,residual_n_pred_samples_tilde_2*2**kr)<get_mse(residual_pred_samples,residual_n_pred_samples_tilde*2**kr):
                    residual_n_pred_samples_tilde=np.copy(residual_n_pred_samples_tilde_2   )         
                #"""
                
                
                #print("code",len(code),ntot-nx+11)
                signal_pred_samples_rec_opt=signal_pred_samples_tilde_opt+residual_n_pred_samples_tilde_opt*2**kr_opt
                signal_pred_samples_rec=signal_pred_samples_tilde+residual_n_pred_samples_tilde*2**kr
                
                MSE_reel_tot_opt[w,nx]=get_mse(signal_pred_samples[w], signal_pred_samples_rec_opt)
                MSE_reel_tot[w,nx]=get_mse(signal_pred_samples[w], signal_pred_samples_rec)
                
                
                
                MSE_reel_r_opt[w,nx]=get_mse(residual_pred_samples_opt,residual_n_pred_samples_tilde_opt*2**(kr_opt))
                MSE_reel_r[w,nx]=get_mse(residual_pred_samples,residual_n_pred_samples_tilde*2**(kr))
                
        
                error_q_opt=signal_pred_samples_hat[w]-signal_pred_samples_tilde_opt
                gamma0_eq_opt=np.mean(error_q_opt**2)
                gamma1_eq_opt=np.sum(error_q_opt[0:N-1]*error_q_opt[1:N])/(N)
                """
                if gamma0_eq_opt<gamma1_eq_opt:
                    memoire=gamma0_eq_opt
                    gamma0_eq_opt= gamma1_eq_opt
                    gamma1_eq_opt= memoire
                #gamma1_eq_opt=np.min([np.mean(error_q_opt[0:N-1]*error_q_opt[1:N]),(1-1/N)*gamma0_eq_opt])
                """
                error_q=signal_pred_samples_hat[w]-signal_pred_samples_tilde
                gamma0_eq=np.mean(error_q**2)
                gamma1_eq=np.sum(error_q[0:N-1]*error_q[1:N])/(N-1)
                """
                if gamma0_eq<gamma1_eq:
                    memoire=gamma0_eq
                    gamma0_eq= gamma1_eq
                    gamma1_eq= memoire
                """   
                    
                #gamma1_eq=np.min([np.mean(error_q[0:N-1]*error_q[1:N]),(1-1/N)*gamma0_eq])#/N
                
                """
                if  gamma0_eq_opt< gamma1_eq_opt:
                    print(gamma0_eq_opt)
                    print(gamma1_eq_opt)
                    autocorrelation(signal_pred_samples_hat[w]-signal_pred_samples_tilde_opt,True)[0:2]
                """    
                            
                
                gamma0_reel_opt[w,nx]=gamma0_eq_opt
                gamma0_reel[w,nx]=gamma0_eq
                
                gamma1_reel_opt[w,nx]=gamma1_eq_opt
                gamma1_reel[w,nx]=gamma1_eq
                
                
        
                #MSE_model_tot_opt[w,nx]=allocation_pred_samples.MSE_theoritical_pred_samples([*al_opt_int[nx]]+[(ntot-np.sum(al_opt_int[nx]))/N],w_theta_pred_samples,gamma0_em[w],gamma1_em[w],gamma0_q[w],gamma1_q[w])
                #MSE_model_tot[w,nx]=allocation_pred_samples.MSE_theoritical_pred_samples([*al_int[nx]]+[(ntot-np.sum(al_int[nx]))/N],w_theta_pred_samples,gamma0_em[w],gamma1_em[w],gamma0_q[w],gamma1_q[w])
                
                MSE_model_tot_opt[w,nx]=allocation_pred_samples.MSE_theoritical_pred_samples([*al_opt_float[nx]]+[(ntot-np.sum(al_opt_float[nx]))/N],w_theta_pred_samples,gamma0_em[w],gamma1_em[w],gamma0_q[w],gamma1_q[w])
                MSE_model_tot[w,nx]=allocation_pred_samples.MSE_theoritical_pred_samples([*al_float[nx]]+[(ntot-np.sum(al_float[nx]))/N],w_theta_pred_samples,gamma0_em[w],gamma1_em[w],gamma0_q[w],gamma1_q[w])
                
                
         
                
            
                test_error_quantization=0
                if test_error_quantization and w <1 and nx in [0,15,20]:#,20]:
                    
                   
                    """
                    plt.figure(figsize=(8,4), dpi=100)
              
                    plt.plot(t,signal_pred_samples_hat[w]-signal_pred_samples_tilde_opt,lw=2,label="error")
                    plt.xlabel('t (s)')
                    plt.ylabel('Amplitude error q')
                    plt.legend()
                    plt.title("Erreur de quantification pred samples nx={}".format(nx))
                    #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    """
                    autocorrelation(signal_pred_samples_hat[w]-signal_pred_samples_tilde_opt,True)
                 
                    
            
                    plt.figure(figsize=(8,4), dpi=100)
                    if not real :
                        plt.plot(t,signal_noise,lw=2,label=r'$\mathbf{{b}}, MSE={:.8f}$'.format(get_mse(signal_noise,signal_noise*0)))
                    plt.plot(t,signal_pred_samples[w]-signal_pred_samples_hat[w],lw=2,label=r'$\mathbf{{x}}-\mathbf{{x}}^{{m}}\left(\widehat{{\mathbf{{\theta}}}}\right), MSE={:.8f}$'.format(get_mse(signal_pred_samples[w],signal_pred_samples_hat[w])))
                    plt.ylabel('amplitude')
                    plt.xlabel('t (s)')
                    plt.legend()
                    plt.title("Erreur de modélisation pred samples")
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    
                    #autocorrelation(signal_poly-signal_poly_hat,True)
                    
                    
                    plt.figure(figsize=(8,4), dpi=100)
                    plt.plot(tp,signal_previous[w],lw=2,label="previous")
                    plt.plot(t,signal_pred_samples[w],lw=2,label=r'$\mathbf{x}$')
                    plt.plot(t,signal_pred_samples_hat[w],lw=2,label=r'$\mathbf{x}^m(\widehat{\mathbf{\theta}})$')
                    plt.plot(t,signal_pred_samples_tilde,lw=2,label=r'$\mathbf{x}^m(\widetilde{\mathbf{\theta}})$')
                    plt.xlabel('t (s)')
                    plt.ylabel('amplitude')
                    plt.legend()
                    #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show() 
                    print("-------------------------------------")
    
                    print("theta pred          ",[np.round(1000*theta_pred_samples[i])/1000 for i in range (order_pred)])
                    print("theta pred hat      ",[np.round(1000*theta_hat[w][i])/1000 for i in range (order_pred)])
                    print("theta pred tilde opt",[np.round(1000*theta_pred_samples_tilde_opt[i])/1000 for i in range (order_pred)])
                    print("theta pred tilde    ",[np.round(1000*theta_pred_samples_tilde[i])/1000 for i in range (order_pred)])
                    
                    
                    print("al pred float opt ={}, nx={:.2f}, nx target={}".format(al_opt_float[nx],np.sum(al_opt_float[nx]),nx))
                    print("al pred int opt={}".format(*al_opt_int[nx]))
                    
                    print("al pred float ={}, nx={:.2f}, nx target={}".format(al_float[nx],np.sum(al_float[nx]),nx))
                    print("al pred int opt={}".format(*al_int[nx]))
                    
                    
            if  w<10:# in [0,10,20,30,40,50,60,70,80,100]:
                root,MSE_model_tot_min = allocation_pred_samples.get_nx_nr_constraint_bit_pred_samples(w_theta_pred_samples,eta,signal_previous[w][1*N:3*N],signal_pred_samples[w]-signal_pred_samples_hat[w],ntot,dtype)   
                #print("root",root)
                #print("MSE_model_tot_min",MSE_model_tot_min)
                          
                plt.figure(figsize=(8,4), dpi=100)   
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot[w][min_:]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot[w][min_:]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot_opt[w][min_:]),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
                plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot_opt[w][min_:]),'-*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, opt},...,n_{K, opt}\right)\right)$')
                plt.plot(np.sum(root[0:order_pred]),10*np.log10(MSE_model_tot_min),'o',lw=4,label="Theoritical point fine")
                plt.xlabel('nx (bits)')
                plt.ylabel('10 log10 MSE (dB)')
                plt.legend()
                plt.title("Visualisation du log MSE de l'erreur de quantification signal pred samples d'ordre {}, fenêtre={}".format(order_pred,w))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()  
                print("------------------------",np.sum(root[0:order_pred]),10*np.log10(MSE_model_tot_min))
                ###########################   Voir si le débit est bien prédi
                MSE_target=MSE_model_tot_min
                print("MSE target={} dB",10*np.log10(MSE_target))
                nx_nr_find,MSE_model_tot_min_find=allocation_pred_samples.get_nx_nr_constraint_MSE_pred_samples(w_theta_pred_samples,eta,signal_previous[w][N:3*N],signal_pred_samples[w]-signal_pred_samples_hat[w],MSE_target,dtype)
                print("MSE find={} dB",10*np.log10(MSE_model_tot_min_find))
                
                print("nx find ={} b, nr find={} b".format(nx_nr_find[0:len(w_theta_pred_samples)],N*nx_nr_find[-1]),"nx taget={} b, nr target={} b".format(root[0:len(w_theta_pred_samples)],N*root[-1]))
                  
                
        plt.figure(figsize=(8,4), dpi=100)   
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_model_q,axis=0)+np.mean(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$ ')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_qm,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1},...,n_{K}\right)\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_model_q_opt,axis=0)+np.mean(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)+\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_qm_opt,axis=0)),'*',lw=2,label=r'$\sigma_{emq}^{2}\left(\mathbf{x},\left(n_{1, opt},...,n_{K, opt}\right)\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("Visualisation du log MSE moyen de l'erreur de quantification sur des signaux polynomiaux d'ordre {}".format(order_pred))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        Error_log_model_q_model_est=np.abs(10*np.log10(MSE_model_q)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)))[0]
        #print("Error_log_model_q_model_est",Error_log_model_q_model_est,len(Error_log_model_q_model_est))
        
        min_nx=np.min(Error_log_model_q_model_est)
        
       
        #print("min_nx",min_nx)
        nx=np.where(Error_log_model_q_model_est==min_nx)[0][0]
        #print("nx",nx)
        
        Error_log_model_q_model_est_opt=np.abs(10*np.log10(MSE_model_q_opt)-10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)))[0]
        min_opt=np.min(Error_log_model_q_model_est_opt)
        nx_opt=np.where(Error_log_model_q_model_est_opt==min_opt)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_q,axis=0)),'*',lw=2,label=r'$\sigma_{q reel}^{2}\left((n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_model_q,axis=0)),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1},...,n_{K}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_q_opt,axis=0)),'*',lw=2,label=r'$\sigma_{q reel}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_model_q_opt,axis=0)),lw=2,label=r'$\sigma_{eq,m}^{2}\left(n_{1, opt},...,n_{K, opt}\right)$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.ones(nx_max)*np.mean(MSE_model,axis=0)),lw=2,label=r'$\sigma_{em}^{2}\left(\mathbf{x}\right)$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("nx opt={}, nx={}".format(nx_opt,nx))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
            
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_r_opt,axis=0)),lw=2,label=r'$\sigma_{r,opt}^{2}$')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(MSE_reel_r,axis=0)),lw=2,label=r'$\sigma_{r}^{2}$')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("nx opt={}, nx={}".format(nx_opt,nx))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        
        
        
        
        plt.figure(figsize=(8,4), dpi=100) 
        #plt.plot([i for i in range(nx_max)],np.log(MSE_model_q_opt),lw=2,label=r'MSE model')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_model_opt,axis=0)),lw=2,label=r'gamma0 model')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma1_model_opt,axis=0)),lw=2,label=r'gamma1 model')
        #plt.plot([i for i in range(nx_max)],np.log(np.mean(MSE_reel_q_opt,axis=0)),'-*',lw=2,label=r'MSE reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0)),'-*',lw=2,label=r'gamma0 reel')
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma1_reel_opt,axis=0)),'-*',lw=2,label=r'gamma1 reel')
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()     
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma0_model_opt,axis=0))-10*np.log10(np.mean(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.mean(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma1_model_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel(' gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_model_opt,axis=0))-10*np.log10(np.median(gamma0_reel_opt,axis=0)),"-*",lw=2,label=r'gamma0 model / gamma0 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(np.median(gamma0_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma1_model_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)),"-*",lw=2,label=r'gamma1 model / gamma1 reel median, mean={:.2f})'.format(np.mean(10*np.log10(gamma1_model_opt)-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel(' gamma model / real (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
          
        
        plt.figure(figsize=(8,4), dpi=100) 
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma0_model_opt,axis=0)+10**(-10))-10*np.log10(np.mean(gamma1_model_opt,axis=0)+10**(-10)),"-*",lw=2,label=r'gamma0 model / gamma1 model, mean={:.2f}'.format(np.mean(10*np.log10(gamma0_model_opt)-10*np.log10(gamma1_model_opt))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.mean(gamma0_reel_opt,axis=0)+10**(-10))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)+10**(-10)),"-o",lw=2,label=r'gamma0 reel mean / gamma1 reel mean, mean={:.2f}'.format(np.mean(10*np.log10(np.mean(gamma0_reel_opt,axis=0))-10*np.log10(np.mean(gamma1_reel_opt,axis=0)))))
        plt.plot([i for i in range(nx_max)],10*np.log10(np.median(gamma0_reel_opt,axis=0)+10**(-10))-10*np.log10(np.median(gamma1_reel_opt,axis=0)+10**(-10)),"-*",lw=2,label=r'gamma0 reel modian / gamma1 reel modian, mean={:.2f}'.format(np.mean(10*np.log10(np.median(gamma0_reel_opt,axis=0))-10*np.log10(np.median(gamma1_reel_opt,axis=0)))))
        plt.xlabel('nx (bits)')
        plt.ylabel('gamma0 / gamma1 (dB)')
        plt.legend()
        #plt.title("Visualisation du log de MSE  due à la quantification sur des signaux de tests sinusoïdaux")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
         
              
    
        ################################################################# mean MSE tot
    
        MSE_model_tot_opt_median=np.median(MSE_model_tot_opt,axis=0)
        min_opt_model_median=np.min(MSE_model_tot_opt_median)
        #print("min_opt_model_median",min_opt_model_median)
        nx_opt_model_median=np.where(MSE_model_tot_opt_median == min_opt_model_median)[0][0]
        
        MSE_model_tot_opt_mean=np.mean(MSE_model_tot_opt,axis=0)
        min_opt_model_mean=np.min(MSE_model_tot_opt_mean)
        nx_opt_model_mean=np.where(MSE_model_tot_opt_mean == min_opt_model_mean)[0][0]
        
        
        
    
        MSE_model_tot_median=np.median(MSE_model_tot,axis=0)
        min_model_median=np.min(MSE_model_tot_median)
        nx_model_median=np.where(MSE_model_tot_median == min_model_median)[0][0]
    
        MSE_model_tot_mean=np.mean(MSE_model_tot,axis=0)
        min_model_mean=np.min(MSE_model_tot_mean)
        nx_model_mean=np.where(MSE_model_tot_mean == min_model_mean)[0][0]
        
            
        
        
    
        MSE_reel_tot_opt_median=np.median(MSE_reel_tot_opt,axis=0)
        min_opt_reel_median=np.min(MSE_reel_tot_opt_median)
        nx_opt_reel_median=np.where(MSE_reel_tot_opt_median == min_opt_reel_median)[0][0]
        
        MSE_reel_tot_opt_mean=np.mean(MSE_reel_tot_opt,axis=0)
        min_opt_reel_mean=np.min(MSE_reel_tot_opt_mean)
        nx_opt_reel_mean=np.where(MSE_reel_tot_opt_mean == min_opt_reel_mean)[0][0]
        
        
        
    
        MSE_reel_tot_median=np.median(MSE_reel_tot,axis=0)
        min_reel_median=np.min(MSE_reel_tot_median)
        nx_reel_median=np.where(MSE_reel_tot_median == min_reel_median)[0][0]
        
        MSE_reel_tot_mean=np.mean(MSE_reel_tot,axis=0)
        min_reel_mean=np.min(MSE_reel_tot_mean)
        nx_reel_mean=np.where(MSE_reel_tot_mean == min_reel_mean)[0][0]
        
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot_mean)[min_:],lw=2,label='Mean MSE model uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot_mean)[min_:],'-*',lw=2,label='Mean MSE real uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot_opt_mean)[min_:],lw=2,label='Mean MSE model adapt')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot_opt_mean)[min_:],'-*',lw=2,label='Mean MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("min m opt={:.1f}, min r opt={:.1f}, min m={:.1f}, min r={:.1f}, nx m opt={}, nx r opt={}, nx m={}, nx r={}".format(10*np.log10(min_opt_model_mean),10*np.log10(min_opt_reel_mean),10*np.log10(min_model_mean),10*np.log10(min_reel_mean),nx_opt_model_mean,nx_opt_reel_mean,nx_model_mean,nx_reel_mean))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        
        plt.figure(figsize=(8,4), dpi=100)  
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot_median[min_:]),lw=2,label='Median MSE model uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot_median[min_:]),'-*',lw=2,label='Median MSE real uniform')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_model_tot_opt_median[min_:]),lw=2,label='Median MSE model adapt')
        plt.plot([i for i in range(min_,nx_max)],10*np.log10(MSE_reel_tot_opt_median[min_:]),'-*',lw=2,label='Median MSE real adapt')
        plt.xlabel('nx (bits)')
        plt.ylabel('MSE (dB)')
        plt.legend()
        plt.title("min m opt={:.2f} , min r opt={:.2f}, min m={:.1f}, min r={:.1f}, nx m opt={}, nx r opt={}, nx m={}, nx r={}".format(10*np.log10(min_opt_model_median),10*np.log10(min_opt_reel_median),10*np.log10(min_model_median),10*np.log10(min_reel_median),nx_opt_model_median,nx_opt_reel_median,nx_model_median,nx_reel_median))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        """
        print("signal reel w={}".format(fen))
        curve_tex(t,v1[fen*N:fen*N+N]/1000,0)    
        
        print("signal reel w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_pred_samples, signal_pred_samples*0)/MSE_reel_tot_opt_mean),0)    
        print("signal model w={}".format(fen))
        curve_tex([100*(1-(ntot-i)/ntot) for i in range(nx_max)],10*np.log10(get_mse(signal_pred_samples, signal_pred_samples*0)/MSE_model_tot_opt_mean),0)    
        """    
    
    


    

