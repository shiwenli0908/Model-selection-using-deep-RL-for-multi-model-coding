# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:40:12 2023

@author: coren
"""



########################################################

#allocation de bits entre modèle et résidu en supposant que le résidu est gaussien


import numpy as np
import matplotlib.pyplot as plt


from Bits_allocation import Allocation_sin,Allocation_poly,Allocation_pred_samples
from codage_model import Model_Encoder
from Models import Model_sin,Model_poly,Model_pred_samples
 
from Measures import get_snr,get_quality
import random

class Allocation_sin_bx_br(Allocation_sin):
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_sin_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_sin_bx_br=verbose_Allocation_sin_bx_br
        super().__init__()
        
        
        

    def get_eqmp_e_sin(self,SEemp,bx,br,m_theta_sin,w_theta_sin):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        bx : int
            Nombre de bits servant à coder le modèle
        br : int
            nombre de bits servant à coder le résidu

        Returns
        -------
        Fonction qui détermine 'eqmp':  l'erreur introduite par le modèle et la quantification de ces paramètres et "e" l'erreur totale entre le signal initiale et le signal reconstruit sur bx +br bits

        """
    
        al_sin=self.get_allocation_sin(bx,m_theta_sin,w_theta_sin,dtype='float')# détermination de l'allocation optimale de bits pour le modèle

        delta=[w_theta_sin[k]*2**(-al_sin[k]) for k in range(3)]

        SEeq=(1*self.N/24)*(delta[0]**2+(w_theta_sin[0]**2/12+m_theta_sin[0]**2)*((4/3)*((np.pi**2*self.N**2)/self.fs**2)*delta[1]**2+delta[2]**2))

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        return SEeqmp,SEe 
            
        
    def get_sin_bx_br(self,SEemp,quality,m_theta_sin,w_theta_sin):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        btot : int
            nombre de bits tot

        Returns
        -------
        Fonction qui détermine bx et br théorique en testant les btot repartition de bits (bx,br)
        """
        SEeqmp=[]
        SEe=[]
        bx=[]
        br=[]
    
        quality_test=np.infty
        
        bx_br_opt=np.infty
        bx_opt=0
        
        
        for bx_test in range(0,64):
            br_test=0
            while 1 :
                SEeqmp_test,quality_test=self.get_eqmp_e_sin(SEemp,bx_test,br_test,m_theta_sin,w_theta_sin)
                #print("bx",bx_test,"br",br_test,"q_test>q target : {:.3f} > {:.3f} = {}".format(quality_test,quality,quality_test>quality))
                
                #print(bx_test,br_test,np.sqrt(quality_test/self.N))
                if np.sqrt(quality_test/self.N)<=quality:
                    break
                br_test+=1
                
            if bx_test+br_test<bx_br_opt:
                bx_opt=bx_test
                bx_br_opt=bx_test+br_test
            #else : 
            #    break
           
            bx.append(bx_test)  
            SEeqmp.append(SEeqmp_test)
            SEe.append(quality_test)
            br.append(br_test)
            
       

        if self.verbose_Allocation_sin_bx_br:
            
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(bx,np.array(bx)+np.array(br),lw=2,label='nx+nr')
            plt.xlabel('nx')
            plt.ylabel('Total bits budget')
            plt.legend()
            plt.title('RMSE max = {:.2f} V, nx opt = {}'.format(quality,bx_opt))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()    
               
       
        #print(bx)  
        #print(br)
        
        return bx_opt,bx,br,SEeqmp,SEe
    


class Allocation_poly_bx_br(Allocation_poly):
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_poly_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_poly_bx_br=verbose_Allocation_poly_bx_br
        super().__init__()
        

    def get_eqmp_e_poly(self,SEemp,bx,br,w_theta_poly):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        bx : int
            Nombre de bits servant à coder le modèle
        br : int
            nombre de bits servant à coder le résidu

        Returns
        -------
        Fonction qui détermine 'eqmp':  l'erreur introduite par le modèle et la quantification de ces paramètres et "e" l'erreur totale entre le signal initiale et le signal reconstruit sur bx +br bits

        """
        order=len(w_theta_poly)-1
        al_poly=self.get_allocation_poly(bx,w_theta_poly,dtype='float')# détermination de l'allocation optimale de bits
        #al_poly=[bx/(order+1)]*(order+1)
        delta=[w_theta_poly[k]*2**(-al_poly[k]) for k in range(order+1)]

        SEeq=self.N*np.sum([self.c[k]*delta[k]**2/12 for k in range(order+1)])

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        """
        if bx in[0,10,127]:
            print("al_poly={}, bx={}, br={}, SEeqmp={:.2f}, SEe={:.2f}, 2**(-2*br/self.N))={:.4f}".format(al_poly,bx,br,SEeqmp,SEe,2**(-2*br/self.N)))
        
        """
        return SEeqmp,SEe 
            
        
    def get_poly_bx_br(self,SEemp,quality,w_theta_poly):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        btot : int
            nombre de bits tot

        Returns
        -------
        Fonction qui détermine bx et br théorique en testant les btot repartition de bits (bx,br)

        """

        SEeqmp=[]
        SEe=[]
        bx=[]
        br=[]
    
        quality_test=np.infty
        
        bx_br_opt=np.infty
        bx_opt=0
        
        
        for bx_test in range(0*len(w_theta_poly),16*len(w_theta_poly)):
            br_test=0
            while 1 :
                SEeqmp_test,quality_test=self.get_eqmp_e_poly(SEemp,bx_test,br_test,w_theta_poly)
                #print("bx",bx_test,"br",br_test,"q_test>q target : {:.3f} > {:.3f} = {}".format(quality_test,quality,quality_test>quality))
                
                #print(bx_test,br_test,np.sqrt(quality_test/self.N))
                if np.sqrt(quality_test/self.N)<=quality:
                    break
                br_test+=1
                
            if bx_test+br_test<bx_br_opt:
                bx_opt=bx_test
                bx_br_opt=bx_test+br_test
            #else : 
            #    break
           
            bx.append(bx_test)  
            SEeqmp.append(SEeqmp_test)
            SEe.append(quality_test)
            br.append(br_test)
            
            
    
        
        return  bx_opt,bx,br,SEeqmp,SEe
    



class Allocation_pred_samples_bx_br(Allocation_pred_samples):
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_pred_samples_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_pred_samples_bx_br=verbose_Allocation_pred_samples_bx_br
        super().__init__()
        

    def get_eqmp_e_pred_samples(self,SEemp,bx,br,m_theta_pred_samples,w_theta_pred_samples,previous_samples,eta):
        
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        bx : int
            Nombre de bits servant à coder le modèle
        br : int
            nombre de bits servant à coder le résidu

        Returns
        -------
        Fonction qui détermine 'eqmp':  l'erreur introduite par le modèle et la quantification de ces paramètres et "e" l'erreur totale entre le signal initiale et le signal reconstruit sur bx +br bits

        """
        order=len(w_theta_pred_samples)
        al_pred_samples=self.get_allocation_pred_samples(bx,m_theta_pred_samples,w_theta_pred_samples)# détermination de l'allocation optimale de bits
        #print("al_pred_samples",al_pred_samples,bx)
        delta=[w_theta_pred_samples[k]*2**(-al_pred_samples[k]) for k in range(order)]
        
        SEeq=np.sum([np.sum(np.array(previous_samples[self.N-eta-k:2*self.N-eta-k])**2)*delta[k]**2/12 for k in range(order)])

        SEeqmp=SEemp+SEeq
        
        #print(SEeq)
        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        """
        if bx in[0,10,127]:
            print("al_poly={}, bx={}, br={}, SEeqmp={:.2f}, SEe={:.2f}, 2**(-2*br/self.N))={:.4f}".format(al_poly,bx,br,SEeqmp,SEe,2**(-2*br/self.N)))
        
        """
        return SEeqmp,SEe 
            
        
    def get_pred_samples_bx_br(self,SEemp,btot,m_theta_pred_samples,w_theta_pred_samples,x_pre,eta):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        btot : int
            nombre de bits tot

        Returns
        -------
        Fonction qui détermine bx et br théorique en testant les btot repartition de bits (bx,br)

        """
        if btot<=0:
            return 0,0,[],[]
        
        SEeqmp=[0]*btot
        SEe=[0]*btot
        for bx_test in range(btot):
            br_test=btot-bx_test
            
            SEeqmp_,SEe_=self.get_eqmp_e_pred_samples(SEemp,bx_test,br_test,m_theta_pred_samples,w_theta_pred_samples,x_pre,eta)
            
            SEeqmp[bx_test]=SEeqmp_
            SEe[bx_test]=SEe_
            
        bx_opt=SEe.index(np.min(SEe))
        br_opt=btot-bx_opt
        

        if self.verbose_Allocation_pred_samples_bx_br:
            order=len(w_theta_pred_samples)-1
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(SEeqmp),lw=2,label='SEeqmp')
            plt.plot(np.log(SEe),lw=2,label='SEe, bx min={}'.format(bx_opt))
            plt.xlabel('bx')
            plt.ylabel('Magnitude pred_samples order {}'.format(order))
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()    
        
        return bx_opt,br_opt,SEeqmp,SEe


    


# Programme principal
if __name__ == "__main__":
    from Normalize import normalize



    verbose = False
    N=128
    fn=50
    fs=6400
    quality=800

    
    t=np.linspace(0,(N-1)/fs,N)
    w=1
    if w==1:
        quality=100
        x=np.array([230248.27586206896, 230348.27586206896, 229634.4827586207, 228865.5172413793, 227279.3103448276, 225389.6551724138, 222165.5172413793, 219044.8275862069, 215313.7931034483, 210555.1724137931, 205186.2068965517, 199458.62068965516, 193424.1379310345, 186468.9655172414, 179000.0, 171482.75862068965, 163965.5172413793, 155934.4827586207, 147600.0, 138803.44827586206, 129700.0, 120493.10344827586, 110775.86206896552, 100955.1724137931, 90320.68965517242, 79782.75862068965, 69451.72413793103, 58558.620689655174, 47717.24137931035, 36924.137931034486, 25980.689655172413, 14626.896551724138, 3324.310344827586, -7824.827586206897, -19434.48275862069, -30583.793103448275, -41579.31034482759, -52524.137931034486, -63162.06896551724, -74617.24137931035, -85051.72413793103, -95075.86206896552, -105506.89655172414, -115327.58620689655, -125096.55172413793, -134506.89655172414, -143813.7931034483, -152865.5172413793, -160641.37931034484, -168668.9655172414, -176341.37931034484, -182937.93103448275, -189127.58620689655, -194855.1724137931, -200686.2068965517, -205441.37931034484, -209686.2068965517, -213727.58620689655, -217306.89655172414, -220479.3103448276, -223496.55172413794, -225437.93103448275, -227231.0344827586, -228762.06896551725, -229889.6551724138, -230144.8275862069, -229682.75862068965, -228662.06896551725, -227127.58620689655, -225131.0344827586, -222217.24137931035, -219148.27586206896, -215413.7931034483, -210555.1724137931, -205289.6551724138, -199613.7931034483, -193320.6896551724, -186417.24137931035, -178796.55172413794, -171534.4827586207, -163965.5172413793, -155886.2068965517, -147751.72413793104, -138751.72413793104, -129648.27586206897, -120544.8275862069, -110879.31034482758, -101006.89655172414, -90524.13793103448, -79886.20689655172, -69810.3448275862, -58813.793103448275, -47920.68965517241, -36975.862068965514, -25929.655172413793, -14678.275862068966, -3170.896551724138, 7927.241379310345, 19485.51724137931, 30532.41379310345, 41682.75862068965, 52627.58620689655, 63468.96551724138, 74824.13793103448, 85203.44827586207, 95382.75862068965, 105765.5172413793, 115431.03448275862, 125248.27586206897, 134658.62068965516, 143662.06896551725, 152817.24137931035, 160537.93103448275, 168620.6896551724, 176137.93103448275, 182837.93103448275, 189231.0344827586, 194755.1724137931, 200582.75862068965, 205493.10344827586, 209789.6551724138, 213831.0344827586, 217206.89655172414, 220631.0344827586, 223700.0, 225744.8275862069, 227331.0344827586, 228917.24137931035])
    elif w==49:
        quality=800
        x=np.array([1534.3103448275863, 1483.1379310344828, 1534.3103448275863, 1534.3103448275863, 1483.1379310344828, 1432.0, 1329.7241379310344, 1432.0, 1380.8620689655172, 1380.8620689655172, 1432.0, 1380.8620689655172, 1380.8620689655172, 1483.1379310344828, 1329.7241379310344, 1380.8620689655172, 1483.1379310344828, 1329.7241379310344, 1380.8620689655172, 1432.0, 1380.8620689655172, 1483.1379310344828, 1483.1379310344828, 1329.7241379310344, 1534.3103448275863, 1227.448275862069, 1329.7241379310344, 1329.7241379310344, 1278.5862068965516, 1329.7241379310344, 1227.448275862069, 1278.5862068965516, 1278.5862068965516, 1380.8620689655172, 1329.7241379310344, 1329.7241379310344, 1227.448275862069, 1227.448275862069, 1176.3103448275863, 1278.5862068965516, 1329.7241379310344, 1227.448275862069, 1227.448275862069, -83106.89655172414, -124482.75862068965, -170920.6896551724, -137679.3103448276, -189844.8275862069, -127089.6551724138, -197158.62068965516, -215365.5172413793, -157982.75862068965, -211886.2068965517, -191224.1379310345, -206465.5172413793, -220734.4827586207, -181403.44827586206, -228865.5172413793, -220375.8620689655, -208665.5172413793, -223855.1724137931, -219710.3448275862, -240372.41379310345, -222165.5172413793, -218637.93103448275, -239351.72413793104, -222268.9655172414, -225182.75862068965, -223241.37931034484, -213217.24137931035, -225437.93103448275, -200531.0344827586, -195010.3448275862, -198793.10344827586, -172044.8275862069, -170203.44827586206, -150513.7931034483, -141513.7931034483, -139110.3448275862, -113589.6551724138, -110162.06896551725, -95996.55172413793, -84489.6551724138, -76203.44827586207, -56358.620689655174, -54468.96551724138, -40913.793103448275, -30737.241379310344, -23883.793103448275, -12121.034482758621, -10331.034482758621, 1278.5862068965516, 7160.0, 14268.965517241379, 22503.103448275862, 29049.310344827587, 41475.862068965514, 47665.51724137931, 58048.275862068964, 67713.79310344828, 76355.1724137931, 86893.10344827586, 93131.03448275862, 104127.58620689655, 114255.1724137931, 122131.03448275862, 132665.5172413793, 140437.93103448275, 149848.27586206896, 156855.1724137931, 162686.2068965517, 170665.5172413793, 175627.58620689655, 181455.1724137931, 186110.3448275862, 189793.10344827586, 194600.0, 195931.0344827586, 197668.9655172414, 198844.8275862069, 199917.24137931035, 201913.7931034483, 202017.24137931035, 203293.10344827586, 205493.10344827586, 206924.1379310345, 208410.3448275862, 208717.24137931035])
    xn,kx=normalize(x)
    
    
    quality_n=quality*2**(-kx)
    print("quality_n",quality_n)
    
    
    
    #################" test signal 1
    al =Allocation_sin_bx_br(N,fs,verbose)
    model_sin =Model_sin(fn,fs,N,verbose=False)
    

    m_theta=[0.75,fn,0]
    w_theta=[0.5,0.2,2*np.pi]
    
    theta=model_sin.get_theta_sin(xn)
    print("theta",theta)
    xn_rec=model_sin.get_model_sin(t, *theta)
    
    
    MSEemp=get_quality(xn,xn_rec,'MSE')
    print("RMSE model", np.sqrt(MSEemp*2**(2*kx)))
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x,lw=2,label='x')
    plt.plot(t,xn_rec*2**kx,lw=2,label='x rec RMSE ={:.2f} V'.format(np.sqrt(MSEemp*2**(2*kx))))
    plt.xlabel('y')
    plt.ylabel('t')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x-xn_rec*2**kx,lw=2,label='e')
    plt.xlabel('y')
    plt.ylabel('t')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    
    bx_opt,bx,br,SEeqmp,SEe=al.get_sin_bx_br(MSEemp*N,quality_n,m_theta,w_theta)
        
    ntot=np.array(bx)+np.array(br)

    print("nx_sin_model=",[bx[i] for i in range(len(bx))])
    print("ntot_sin_model=",[ntot[i] for i in range(len(ntot))])
        
    plt.figure(figsize=(8,4), dpi=100)
    if w==1 :
        nx_sin= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        ntot_sin= [353, 354, 355, 356, 357, 349, 350, 345, 346, 347, 347, 348, 339, 344, 345, 336, 329, 330, 329, 338, 339, 344, 339, 340, 335, 339, 340, 341, 339, 340, 346, 357, 347, 344, 342, 345, 346, 347, 348, 349, 346, 351, 352, 352, 350]

        plt.plot(nx_sin,ntot_sin,lw=2,label='ntot reel')
        
    plt.plot(bx,np.array(bx)+np.array(br),lw=2,label='ntot model polynomial sin')
    plt.xlabel('nx')
    plt.ylabel('nx+nr')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {} b, min model = {}'.format(quality,bx_opt,np.min(ntot)))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()   
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.sqrt(np.array(SEe)*2**(2*kx)/N),lw=2,label='nx+nr reel')
    plt.xlabel('nx')
    plt.ylabel('Total bits budget')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {}'.format(quality,bx_opt))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(bx,np.sqrt(np.array(SEeqmp)*2**(2*kx)/N),lw=2,label='erreur de quantification')
    plt.xlabel('nx')
    plt.ylabel('eq')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {}'.format(quality,bx_opt))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
 

    






    #"""

    #################" test poly
    al =Allocation_poly_bx_br(N,fs,verbose)
    
    model_poly =Model_poly(fn,fs,N,verbose=False)
    

    order=7
    
    m_theta=[0]*(order+1)
    w_theta=[2]*(order+1)
    
    theta=model_poly.get_theta_poly(xn,order)
    print("theta",theta)
    xn_rec=model_poly.get_model_poly(t, *theta)
    
    
    MSEemp=get_quality(xn,xn_rec,'MSE')
    print("MSE model", MSEemp*2**(2*kx))
    #MSEemp2=get_quality(x,xn_rec*2**kx,'MSE')
    #print("MSEemp2", MSEemp2)
    print("RMSE model", np.sqrt(MSEemp*2**(2*kx)))
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x,lw=2,label='x')
    plt.plot(t,xn_rec*2**kx,lw=2,label='x rec poly {} RMSE ={:.2f} V'.format(order,np.sqrt(MSEemp*2**(2*kx))))
    plt.xlabel('t (s)')
    plt.ylabel("amplitude (V)")
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x-xn_rec*2**kx,lw=2,label='error poly {} RMSE ={:.2f} V'.format(order,np.sqrt(MSEemp*2**(2*kx))))
    plt.xlabel('t (s)')
    plt.ylabel("amplitude (V)")
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    
    bx_opt,bx,br,SEeqmp,SEe=al.get_poly_bx_br(MSEemp*N,quality_n,w_theta)
    ntot=np.array(bx)+np.array(br)
    print("nx_poly7_model=",[bx[i] for i in range(len(bx))])
    print("ntot_poly7_model=",[ntot[i] for i in range(len(ntot))])
    

    plt.figure(figsize=(8,4), dpi=100)
    if w==1 :
        if order==6:
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
            ntot_reel= [357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 517, 518, 525, 526, 527, 479, 480, 444, 445, 450, 451, 452, 463, 464, 447, 448, 452, 453, 454, 454, 455, 464, 465, 458, 405, 406, 421, 422, 427, 442, 441, 442, 443, 402, 393, 382, 389, 389, 375, 376, 359, 360, 374, 375, 379, 378, 374, 385, 387, 382, 381, 385, 389, 390, 388, 393, 386, 387, 385, 402, 403, 390, 382, 390, 390, 392, 393, 411, 398, 399, 401, 409, 409, 410, 411, 408, 410, 410, 410, 410, 411, 416, 421, 420, 420]
        elif order ==7:
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
            ntot_reel= [357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 519, 520, 527, 528, 529, 530, 482, 483, 447, 448, 453, 454, 455, 456, 467, 468, 451, 452, 456, 457, 458, 459, 459, 460, 469, 470, 463, 464, 411, 412, 427, 428, 433, 434, 438, 439, 442, 443, 408, 401, 395, 396, 396, 397, 383, 384, 367, 368, 382, 383, 387, 388, 387, 383, 394, 396, 391, 390, 394, 386, 394, 376, 378, 362, 366, 367, 356, 357, 358, 390, 393, 392, 399, 397, 395, 396, 397, 403, 404, 405, 402, 403, 405]
        plt.plot(nx_reel,ntot_reel,lw=2,label='ntot reel poly {}'.format(order))  
    elif w==49 :   
        if order ==1:
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            ntot_reel= [621, 622, 623, 624, 606, 607, 583, 585, 501, 506, 567, 568, 552, 558, 567, 568, 563, 564, 563, 566, 567, 568, 571, 572, 573, 574, 575, 576, 575, 576, 577, 578]
        elif order==2:    
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
            ntot_reel= [621, 622, 623, 624, 625, 621, 624, 625, 620, 593, 597, 587, 578, 577, 579, 584, 585, 588, 599, 598, 603, 606, 609, 610, 609, 611, 614, 615, 616, 618, 620, 621, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652]
        elif order==3:  
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
            ntot_reel= [621, 622, 623, 624, 625, 626, 627, 623, 626, 627, 628, 623, 596, 600, 560, 575, 628, 630, 598, 598, 594, 595, 604, 605, 610, 611, 611, 610, 609, 611, 612, 611, 622, 626, 626, 627, 627, 628, 623, 627, 630, 631, 632, 633, 634, 635, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652]
        elif order==4:  
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
            ntot_reel= [621, 622, 623, 624, 625, 626, 627, 628, 629, 625, 628, 629, 600, 601, 612, 570, 571, 576, 565, 570, 593, 600, 603, 597, 592, 601, 602, 591, 597, 607, 599, 602, 606, 607, 607, 605, 617, 593, 597, 595, 612, 617, 617, 621, 622, 609, 610, 612, 613, 612, 625, 626, 628, 629, 630, 617, 618, 619, 620, 621, 622, 623, 624, 625]
        elif order==5:  
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
            ntot_reel= [621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 627, 630, 631, 632, 603, 604, 615, 573, 574, 575, 580, 581, 585, 575, 581, 585, 599, 580, 586, 569, 570, 571, 572, 573, 585, 601, 602, 589, 601, 611, 613, 615, 617, 599, 598, 619, 617, 602, 607, 607, 613, 614, 613, 631, 632, 633, 634, 635, 636, 642, 643, 644, 647, 642, 642]
        elif order==6:  
            nx_reel= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
            ntot_reel= [621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 629, 632, 633, 650, 651, 660, 661, 644, 605, 610, 572, 573, 573, 574, 579, 552, 553, 549, 596, 597, 567, 571, 593, 594, 580, 581, 582, 581, 586, 592, 588, 611, 588, 589, 589, 598, 616, 619, 615, 616, 615, 616, 619, 623, 624, 597, 598, 600, 601, 627, 604, 605, 606]
            
        plt.plot(nx_reel,ntot_reel,lw=2,label='ntot reel poly {}'.format(order))  
    
        
    plt.plot(bx,np.array(bx)+np.array(br),lw=2,label='ntot model polynomial ordre {}'.format(order))
    plt.xlabel('nx')
    plt.ylabel('nx+nr')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {} b, min model = {}'.format(quality,bx_opt,np.min(ntot)))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()   
   
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(bx,np.sqrt(np.array(SEe)*2**(2*kx)/N),lw=2,label='RMSE tot')
    plt.xlabel('nx')
    plt.ylabel('RMSE V')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {}'.format(quality,bx_opt))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
 
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(bx,np.sqrt(np.array(SEeqmp)*2**(2*kx)/N),lw=2,label='erreur de quantification')
    plt.xlabel('nx')
    plt.ylabel('eq')
    plt.legend()
    plt.title('RMSE max = {:.2f} V, nx opt = {}'.format(quality,bx_opt))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
 
    
    

    
    
    
    #################################################" deuxième exemple avec fenêtre 25
    #"""

    
    
    
    
    
    
    
    
    
    
    
    