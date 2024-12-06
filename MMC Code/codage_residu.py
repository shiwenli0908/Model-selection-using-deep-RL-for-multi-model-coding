# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:09:09 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
import pywt 


from Antonini import Antonini_Encoder,Antonini_Decoder
from Khan_EZW import Khan_Encoder,Khan_Decoder
from Measures import get_quality,entropy

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Residual_Encoder(Antonini_Encoder,Khan_Encoder):
    def __init__(self,N=128,factor_scale=3):
        
        self.N=N
        M=9
        
        initial_occurrence_first_Antonini=[1,1]
        initial_occurrence_second_Antonini=[1,1,1,1,1]
        Antonini_Encoder.__init__(self,
                                  M,
                                  initial_occurrence_first=initial_occurrence_first_Antonini,
                                  initial_occurrence_second=initial_occurrence_second_Antonini,
                                  adaptive=True,
                                  verbose_AE=False) 
        
        #print(self.initial_occurrence_second)
        
        #initialisation paramètres ondelettes
        self.wave_test = 'coif5'
        self.level =  int(np.ceil(np.log2(N)))
        self.mode= 'periodization'
        
        initial_occurrence_first_Khan=[1,1]
        initial_occurrence_second_Khan=[1,1,1,1]
        Khan_Encoder.__init__(self,
                              level=self.level,
                              wave_test=self.wave_test,
                              M=M,
                              initial_occurrence_first_Khan=initial_occurrence_first_Khan,
                              initial_occurrence_second_Khan=initial_occurrence_second_Khan,
                              adaptive_Khan=True,
                              verbose_KE=False) 
        #print(self.initial_occurrence_second)
        self.factor_scale=factor_scale
        
    def get_r_DCT_BPC_tilde(self,r,metric,quality,n_max):
        ################## Antonini DCT
        if metric !="SNR":
            quality/=self.factor_scale
        coefs_DCT=dct(r/self.N)/self.factor_scale
        
        code_DCT=self.get_code_res_Antonini(coefs_DCT,metric,quality,n_max)
        
        coefs_rec_DCT=self.coefs_rec*self.factor_scale
        """
        if np.max(abs(coefs_DCT))>1:
            print("Warning max coef DCT = {}".format(np.max(abs(coefs_DCT))))
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(coefs_DCT,lw=2,label='coefs')
            plt.plot(self.coefs_rec,lw=2,label='coefs rec')
            plt.xlabel('ind')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title('DCT, Nb bits used / nb bits max = {} / {}'.
                     format(len(code_DCT),n_max))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        """
        
        
        r_rec_DCT=self.get_x_rec_Antonini(coefs_rec_DCT)
        
        return r_rec_DCT,code_DCT
    
    
        
    def get_r_DWT_BPC_tilde(self,r,metric,quality,n_max):
        
        if metric !="SNR":
            quality/=self.factor_scale
        coefs = pywt.wavedec(r, self.wave_test, mode=self.mode, level=self.level)
        
        coefs_DWT = []
        # Pour chaque niveau de décomposition
        for i in range(self.level + 1):
            # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
            coefs_DWT.extend(coefs[i]/(self.level*self.factor_scale))
    
    
        code_DWT=self.get_code_res_Khan(coefs_DWT,metric,quality,n_max)
        
        coefs_rec_DWT=self.coefs_rec*self.factor_scale
        """
        if np.max(abs(np.array(coefs_DWT)))>1:
            print("Warning max coef DWT = {}".format(np.max(abs(np.array(coefs_DWT)))))
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(coefs_DWT,lw=2,label='coefs')
            plt.plot(self.coefs_rec,lw=2,label='coefs rec')
            plt.xlabel('ind')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title('DWT, Nb bits used / nb bits max = {} / {}'.
                     format(len(code_DWT),n_max))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        """
        
        r_rec_DWT=self.get_x_rec_Khan(coefs_rec_DWT)
        
        return r_rec_DWT,code_DWT
        



class Residual_Decoder(Antonini_Decoder,Khan_Decoder):
    def __init__(self,N=128,factor_scale=3):
        
        
        
        M=9
        initial_occurrence_first_Antonini=[1,1]
        initial_occurrence_second_Antonini=[1,1,1,1,1]
        Antonini_Decoder.__init__(self,
                                  N=N,
                                  M=M,
                                  initial_occurrence_first=initial_occurrence_first_Antonini,
                                  initial_occurrence_second=initial_occurrence_second_Antonini,
                                  adaptive=True,
                                  verbose_AD=False) 
        

        
        
        #initialisation paramètres ondelettes
        self.wave_test = 'coif5'
        self.level =  int(np.ceil(np.log2(N)))
        self.mode= 'periodization'
        
        initial_occurrence_first_Khan=[1,1]
        initial_occurrence_second_Khan=[1,1,1,1]
        Khan_Decoder.__init__(self,
                              level=self.level,
                              wave_test=self.wave_test,
                              N=N,
                              M=M,
                              initial_occurrence_first_Khan=initial_occurrence_first_Khan,
                              initial_occurrence_second_Khan=initial_occurrence_second_Khan,
                              adaptive_Khan=True,
                              verbose_KD=False)         
           
        self.factor_scale=factor_scale
    def  get_r_DCT_BPC_tilde(self,code,n_max):
        coefs_rec=self.get_coefs_rec_Antonini(code,n_max)*self.factor_scale
        r_rec=idct(coefs_rec)/2
        
        return r_rec
    
    def get_r_DWT_BPC_tilde(self,code,n_max):
        
        coefs_L=self.get_coefs_rec_Khan(code,n_max)*self.factor_scale
        
        
        
        r_rec=self.get_x_rec_Khan(coefs_L)
        return r_rec
        

# Programme principal
if __name__ == "__main__":
    from Normalize import normalize
    from Models import Model_poly
    
    metric="RMSE"
    quality=0.01 #dB
    unity="V"
    n_max=60 #bits

    N=128 
    fn=50
    fs=6400
    
    factor_scale=2
    t=np.linspace(0,(N-1)/fs,N)
    
    sigma=0.05 # écart type du n_ruit introduit dans le signal test
    
    ###############  test polynôme d'ordre k
    order=8
    theta=np.random.uniform(-0.2,0.2,order+1) #[-0.4, -0.3,  0.2 , 0.05 , 0.8 ,-0.3]#
   
    model_poly=Model_poly(fn,fs,N,verbose=False)

    x_test=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    x_test,_=normalize(x_test)

    """
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x test')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    """
    ############################# encodage résidu sur n_r bits

    l=Residual_Encoder(N,factor_scale)
        
    x_dec_enc_DCT,code_DCT=l.get_r_DCT_BPC_tilde(x_test, metric, quality,n_max)
    
    #print("code=",code)
    print("Nb bits used / Nb bits max = {} / {} bits".format(len(code_DCT),n_max),"{} = {}  {}/ {} ".
          format(metric,get_quality(x_test,x_dec_enc_DCT,metric),quality,unity))
    print("Nb sym codé / Nb sym max = {} / {}".format(l.nb_coefs,l.nb_coefs_max))
    #print(AE.symbol)

    occurrence_first_=l.occurrence_first
    occurrence_second_=l.occurrence_second
    print("Occurrence des symboles des premières passe",occurrence_first_)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second_)
        
        
    occurrence_first=np.array(l.occurrence_first_true)-1
    occurrence_second=np.array(l.occurrence_second_true)-1
    
    if np.sum(occurrence_first)!=0:
        p_first=occurrence_first/np.sum(occurrence_first)
        H_first=entropy(p_first)
    else:
        p_first=occurrence_first
        H_first=0
        
    if np.sum(occurrence_second)!=0:
        
        p_second=occurrence_second/np.sum(occurrence_second)
        H_second=entropy(p_second)
    else:
        p_second=occurrence_second
        H_second=0
        
    print("Occurrence des symboles des premières passe",occurrence_first)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second)
    
    print("H first = {:.2f} bits".format(H_first))
    print("H second = {:.2f} bits".format(H_second))

    H_tot=np.sum(occurrence_first)*H_first+np.sum(occurrence_second)*H_second
    print("H tot={:.2f} bits".format(H_tot))
    
    print("DCT+BPC","{} = {:.5f} / {} {}".format(metric,get_quality(x_test,x_dec_enc_DCT,metric),quality,unity),"len(code)={}".format(len(code_DCT)))
    
    x_dec_enc_DWT,code_DWT=l.get_r_DWT_BPC_tilde(x_test, metric, quality,n_max)
    
    print("DWT+BPC","{} = {:.5f} / {} {}".format(metric,get_quality(x_test,x_dec_enc_DWT,metric),quality,unity),"len(code)={}".format(len(code_DWT)))
    

    ######################### test best residual dec

    
    l_dec=Residual_Decoder(N,factor_scale)
    
    x_dec_DCT=l_dec.get_r_DCT_BPC_tilde( code_DCT,n_max)
    x_dec_DWT=l_dec.get_r_DWT_BPC_tilde( code_DWT,n_max)
         
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x')
    plt.plot(t,x_dec_enc_DCT,lw=2,label='x dec encoder, {}_enc = {:.5f}/ {} {}'.
             format(metric,get_quality(x_test,x_dec_enc_DCT,metric),quality,unity))
    plt.plot(t,x_dec_DCT,lw=2,label='x dec decoder,  {}_dec = {:.5f} / {} {}'.
             format(metric,get_quality(x_test,x_dec_DCT,metric),quality,unity))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Résidu reconstruit DCT")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
                
   
         
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x')
    plt.plot(t,x_dec_enc_DWT,lw=2,label='x dec encoder, {}_enc = {:.5f}/ {} {}'.
             format(metric,get_quality(x_test,x_dec_enc_DWT,metric),quality,unity))
    plt.plot(t,x_dec_DWT,lw=2,label='x dec decoder,  {}_dec = {:.5f} / {} {}'.
             format(metric,get_quality(x_test,x_dec_DWT,metric),quality,unity))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Résidu reconstruit DWT")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
                
    
    
    print("Nb bits used / Nb bits max = {} / {} bits".format(len(code_DWT),n_max),"{} = {} / {} {}".
          format(metric,get_quality(x_test,x_dec_DWT,metric),quality,unity))
    print("Nb sym codé / Nb sym max = {} / {}".format(l.nb_coefs,l.nb_coefs_max))
    
  
    
    occurrence_first_Khan_=l.occurrence_first_Khan
    occurrence_second_Khan_=l.occurrence_second_Khan
    print("Occurrence des symboles des premières passe",occurrence_first_Khan_)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second_Khan_)
        
        
    occurrence_first_Khan=np.array(l.occurrence_first_true_Khan)-1
    occurrence_second_Khan=np.array(l.occurrence_second_true_Khan)-1
    
    if np.sum(occurrence_first_Khan)!=0:
        
        p_first=occurrence_first_Khan/np.sum(occurrence_first_Khan)
        H_first=entropy(p_first)
    else:
        p_first=occurrence_first_Khan
        H_first=0
        
    if np.sum(occurrence_second_Khan)!=0:
        
        p_second=occurrence_second_Khan/np.sum(occurrence_second_Khan)
        H_second=entropy(p_second)
    else:
        p_second=occurrence_second_Khan
        H_second=0
        
    print("Occurrence des symboles des premières passe",occurrence_first_Khan)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second_Khan)
    
    print("H first = {:.2f} bits".format(H_first))
    print("H second = {:.2f} bits".format(H_second))

    H_tot=np.sum(occurrence_first_Khan)*H_first+np.sum(occurrence_second_Khan)*H_second
    print("H tot={:.2f} bits".format(H_tot))
   
    
   
    """ 
    #################################  test on large database
    
    import copy
    
    V_max=90000#	 max voltage
    fs = 6400  # sampling frequency of the generated signals
    fn = 50  # nominal frequency of the electrical signals
    delta=18.310 # quantization step size
    t=np.linspace(0,(N-1)/fs,N) #vector time
  
    DATA_u= np.load('DATA_u.npz')['DATA_u'] #  Load DATA_u from the npz file
    
    #DATA_i_load = np.load('DATA_i.npz')['DATA_i'] # Load DATA_i from the npz file
    ## test 
    #print("DATA_S_load",np.shape(DATA_S))
    print("DATA_u",np.shape(DATA_u))
    #print("DATA_i_load",np.shape(DATA_i))
    
    N_test=100
    x_test=copy.deepcopy(DATA_u[0:N_test]*delta/V_max)
    
    
    for w_test in range(N_test):
        x_test[w_test],_=normalize(x_test[w_test])
    
    print("x_test",np.shape(x_test))
    for w in range(5):
        fig=plt.figure(figsize=(8,5),dpi=100)
        plt.plot(t[0:N],x_test[w],lw=2)
        plt.xlabel('t (s)')
        plt.ylabel('Voltage (V)')
        plt.title("w={}, mean={:.2f} V, std={:.2f} V".format(w,np.mean(x_test[w]),np.std(x_test[w])))
        plt.grid( which='major', color='#666666', linestyle='-')
        #plt.legend()
        plt.minorticks_on()
      
    quality=0
    metric="MSE"
    n_max=160
    R=[]
    SNR=[]
    for w in range(100):
        x_dec,code=l.get_r_DWT_BPC_tilde(x_test[w], metric, quality,n_max)
        
        SNR.append(get_snr(x_test[w], x_dec))
        R.append(len(code))
    
    print("SNR mean",np.mean(SNR))
    print("R mean",np.mean(R))
    """  
    
 





