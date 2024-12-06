# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: presvôts

La fonction pour calculer la valeur de SNR en indiquant le choix de modele

n_x = 10 bits par defaut, on peut changer la valeur de n_x pour les choix differents de n_x

id_residual=1 par defaut

"""

from MMC_test_model import Encode_one_window,Decode_one_window
from Measures import get_snr,get_rmse
import numpy as np
#import matplotlib.pyplot as plt

def model_snr(x, state, action, num_phase):

    N=128 # size of window
    fn=50 # nominal frequency
    fs=6400 # samples frequency
    
    """
    # Set of models
    """
    
    Model_used = {} # set of models
    
    ######################################################### None
    id_model=0
    Model_used[id_model]={"name":"none","family":"none"}
    id_model+=1 # id model     
            
    ######################################################### Sinusoidals family
    
    # a priori distribution for theta sin 1
    m_theta_sin=[0.75,fn,0] # mean
    w_theta_sin=[0.5,0.4,2*np.pi] # width
    
    Model_used[id_model]={"name":"sin-1","family":"sin","m theta": m_theta_sin,"w theta": w_theta_sin}
    id_model+=1
    
    ######################################################### Polynomials family
    
    sigma_theta=np.array([0.23,0.29,0.36,0.23,0.12,0.06,0.04,0.03,0.03,0.03,0.02])
    for order in [0,1,2,3,4,5,6]: 
        # a priori distribution for theta poly
        m_theta_poly=[0]*(order+1) # mean
        w_theta_poly=sigma_theta[0:order+1]*7 # width
        
        Model_used[id_model]={"name":"poly-{}".format(order),"family":"poly","order": order,"m theta": m_theta_poly,"w theta": w_theta_poly}   
        id_model+=1 
    
    ######################################################### Predictives samples family
    for eta in range(0,2): 
        for order in range(1,3):
            # a priori distribution for theta pred samples
            m_theta_pred_samples=[0]*order #mean
            w_theta_pred_samples=[1]*order #width
            
            Model_used[id_model]={"name":"samp.-{}-{}".format(order,eta),"family":"pred samples","model used":0, "order" : order,"eta" :eta , "m theta" : m_theta_pred_samples,"w theta" : w_theta_pred_samples}    
            id_model+=1    
    
    ######################################################### Predictives parametric family
    for factor in [2,4,10]:
        
        id_previous_model=3 # model used in the previous window
        
        # a priori distribution for theta pred parametric
        m_theta_pred_para=Model_used[id_previous_model]["m theta"] #mean
        w_theta_pred_para=np.array(Model_used[id_previous_model]["w theta"])/factor #width
        
        Model_used[id_model]={"name":"para.-{}".format(factor),"family":"pred para", "model used":id_previous_model, "factor" : factor, "m theta" : m_theta_pred_para,"w theta":w_theta_pred_para} 
        id_model+=1
        
    
    """
    # Set of method of residual compression
    """   
        
    Residual_used = {}
    
    Residual_used[0]={"name":"none"}
    Residual_used[1]={"name":"DCT+BPC"}
    Residual_used[2]={"name":"DWT+BPC"}
        
    """
    # Initialisation MMC
    """  
        
    #initialisation inputs
    
    #id_signal=1 #fault event from database
    nb_w=100 # number of window per signals
    #nb_min=0 # start window in the signal
    
    nb_phase=1#len(x) # number of phase encode 
    
    #################################### visualisation of signal
    #t=np.linspace(0,(nb_w)*(N-1)*(1/fs),nb_w*N)
    
    #initialisation outputs 
    
    x_rec=np.zeros((nb_phase,N*nb_w)) # signal rec (model+residual)
    x_model=np.zeros((nb_phase,N*nb_w)) # model rec 
    x_residual=np.zeros((nb_phase,N*nb_w)) # residual rec
    
    SNR=np.zeros((nb_phase,nb_w)) # SNR rec (model+residual)
    SNR_m=np.zeros((nb_phase,nb_w)) #SNR model
    SNR_r=np.zeros((nb_phase,nb_w)) # SNR residual
    
    RMSE=np.zeros((nb_phase,nb_w)) # RMSE rec (model+residual)
    
    R_m=np.zeros((nb_phase,nb_w)) # rate model
    R_r=np.zeros((nb_phase,nb_w)) # rate residual
    R_h=np.zeros((nb_phase,nb_w)) # rate header
    R_unused=np.zeros((nb_phase,nb_w))
     
    M=np.zeros((nb_phase,nb_w)) # index of models used
    L=np.zeros((nb_phase,nb_w)) # index of models used
        
    # initialise MMC per phase
    dict_MMC_enc={}
    dict_MMC_dec={}
    for k in range(nb_phase):
        dict_MMC_enc[k]=Encode_one_window(fn,fs,N,Model_used,Residual_used)
        dict_MMC_dec[k]=Decode_one_window(fn,fs,N,Model_used,Residual_used)
        
    """
    #  MMC
    """  
    
    #id_model=4
    id_model = action
    
    '''
    if action == 0:
        id_model = 1
    elif action == 1:
        id_model = 4
    '''
    
    id_residual=1
    n_max=128
    n_x=10
    
    w = state
    
    for phase in range(nb_phase):
        #for w in range(nb_w): 
            
        x_test=x[phase][w*N:(w+1)*N]
        
        if w==0:
            x_previous=np.zeros(N*2) # memory of window previously encoded
        else : 
            if w==1:
                x_previous=np.zeros(N*2) 
                x_previous[N:2*N]=x_rec[phase][(w-1)*N:w*N] 
            else :
                x_previous=x_rec[phase][(w-2)*N:w*N]   
                
        # encode window x_test    
        dict_MMC_enc[phase].MMC_enc(x_test,x_previous,id_model,id_residual,n_x,n_max)
        #SNRenc=get_snr(x[phase][w*N:(w+1)*N],dict_MMC_enc[phase].x_rec_enc)
        
        """
        # decode window x_test
        dict_MMC_dec[phase].MMC_dec(dict_MMC_enc[phase].code,x_previous,n_max)
        SNRdec=get_snr(x[phase][w*N:(w+1)*N], dict_MMC_dec[phase].x_rec_dec)

        if SNRenc!=SNRdec:
            print(SNRenc,SNRdec)
            print("ERREUR of codage!!!!!!!!!!!!!!!!!!!!!!!!!!")
        """
        # save outputs
        
        x_model[phase][w*N:(w+1)*N]=dict_MMC_enc[phase].x_model_enc
        x_residual[phase][w*N:(w+1)*N]=dict_MMC_enc[phase].x_residual_enc
        x_rec[phase][w*N:(w+1)*N]=x_model[phase][w*N:(w+1)*N]+x_residual[phase][w*N:(w+1)*N]
        
        SNR[phase][w]=get_snr(x[phase][w*N:(w+1)*N],x_rec[phase][w*N:(w+1)*N])
        SNR_m[phase][w]=get_snr(x[phase][w*N:(w+1)*N],x_model[phase][w*N:(w+1)*N])
        SNR_r[phase][w]=get_snr(x[phase][w*N:(w+1)*N]-x_model[phase][w*N:(w+1)*N],x_residual[phase][w*N:(w+1)*N])

        RMSE[phase][w]=get_rmse(x[phase][w*N:(w+1)*N],x_rec[phase][w*N:(w+1)*N]) 

        R_m[phase][w]=dict_MMC_enc[phase].nx_enc
        R_r[phase][w]=dict_MMC_enc[phase].nr_enc
        R_h[phase][w]=dict_MMC_enc[phase].n_kx_enc+dict_MMC_enc[phase].n_kr_enc+dict_MMC_enc[phase].nm+dict_MMC_enc[phase].nl+dict_MMC_enc[phase].n_nx_enc
        R_unused[phase][w]=n_max-R_m[phase][w]-R_r[phase][w]-R_h[phase][w]
        
        M[phase][w]=dict_MMC_enc[phase].id_model_enc
        L[phase][w]=dict_MMC_enc[phase].id_residual_enc
        
        '''
        print(f"w={w+1:3}, ph:{phase+1:1}",end='')
        print(f", n_tot={len(dict_MMC_enc[phase].code):3}",end='')
        print(f", SNR={SNR[phase][w]:5.2f} dB",end='')
        print(f", RMSE={RMSE[phase][w]:6.2f} V",end='')
        print(f", m={dict_MMC_enc[phase].m_enc:10}, l={dict_MMC_enc[phase].l_enc:7}",end='')
        print(f", SNR_m={SNR_m[phase][w]:4.1f} dB, SNR_r={SNR_r[phase][w]:4.1f} dB",end='')
        print(f", nh=n_kx+n_kr+nm+nl+n_nx={dict_MMC_enc[phase].n_kx_enc:1}+{dict_MMC_enc[phase].n_kr_enc:1}+{dict_MMC_enc[phase].nm_enc:1}+{dict_MMC_enc[phase].nl_enc:1}+{dict_MMC_enc[phase].n_nx_enc:1}={dict_MMC_enc[phase].n_kx_enc+dict_MMC_enc[phase].n_kr_enc+dict_MMC_enc[phase].nm_enc+dict_MMC_enc[phase].nl_enc+dict_MMC_enc[phase].n_nx_enc:2} b",end='')
        print(f", nx={dict_MMC_enc[phase].nx_enc} b, nr={dict_MMC_enc[phase].nr_enc:4} b",end='') 
        print(f", kx={dict_MMC_enc[phase].kx_enc:3}, kr={dict_MMC_enc[phase].kr_enc:2}",end="")
        print('') 
        '''
        
    return SNR[num_phase][w]
  

    
