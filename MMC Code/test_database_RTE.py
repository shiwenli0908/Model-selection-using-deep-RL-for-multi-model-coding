# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: presvotscor
"""


#from  codage_model import Model_Encoder,Model_Decoder
#from  codage_residu import Residual_Encoder,Residual_Decoder
#from  Allocation_two_stages import Allocation_sin_bx_br,Allocation_poly_bx_br,Allocation_pred_samples_bx_br

#from Quantization import Quantizer
#from Normalize import normalize
import time
import cProfile

#########################" attention residu sup uniquement sur DCT
#from MMC_bruteforce import Encode_one_window,Decode_one_window
#from MMC_sub_optimal_bruteforce import Encode_one_window,Decode_one_window
#from MMC_dicho import Encode_one_window,Decode_one_window
from Model_Rate_Distortion import Encode_one_window,Decode_one_window
from Measures import get_snr,get_rmse


import numpy as np
import matplotlib.pyplot as plt

#from tree_phases import transform_tree,inverse_transform_tree,allocation_optimal,est_equilibre

#from Bits_allocation import Allocation
#from Models import Model_sin

#import copy

verbose = False





"""
# Open the data
"""

from get_test_signal import get_RTE_signal,get_EPRI_signal,get_RTE_signal_new

#"""
N=128 # size of window
fn=50 # nominal frequency
fs=6400 # samples frequency


metric="RMSE"
quality_target=200 #dB
unity="dB"
bmax=1000 #bits

#bm=128
nb_w=100#int(100/factor) # number of window to code
nb_phase=1
nb_signal=1
   
   
M=np.zeros((nb_signal,nb_phase,nb_w))
L_pred_para=np.zeros((nb_signal,nb_phase,nb_w))
    
for signal in range(nb_signal):
    v1,v2,v3,i1,i2,i3=get_RTE_signal_new(signal)
    #initialisation MMC
    x=[v1[0:N*nb_w],v2[0:N*nb_w],v3[0:N*nb_w]]#,,i2[0:N*100],i3[0:N*100]]#+list(v2[0:N*100])+list(v3[0:N*100])#+list(i1[0:N*100])+list(i2[0:N*100])+list(i3[0:N*100])#copie du signal à compresser
   
    
    
    t=np.linspace(0, (nb_w-1)*N*(1/fs),nb_w*N)
    #"""
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(t,v1[0:len(t)]/1000,lw=1,label="v1")
    plt.plot(t,v2[0:len(t)]/1000,lw=1,label="v2")
    plt.plot(t,v3[0:len(t)]/1000,lw=1,label="v3")
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
 
    
    
    """
    ###### Models in the set of models
    """
    
    nb_max_bit_theta=12
    nb_max_bit_theta_pred_para=12
    
    Model_used = {}
         
            
    # Sinusoidals family
    id_model=0
    
    
    
    Model_used[id_model]={"name":"sin-1","family":"sin","m theta": [0.75,fn,0],"w theta": [0.5,0.4,2*np.pi],"b bx": int(np.ceil(np.log2(3*nb_max_bit_theta)))}
    id_model+=1
    
    
    
    
    # Polynomials family
    w_theta_poly_2=np.array([2/6,2/6,2/6,0.23108777,0.1265591,0.05911634,0.0368878,0.02813623,0.02501831,0.02199064,0.0220293 ,0.01964315])*6
    
    for order in [0,1,2,3,4]:#,5,6,7]:  
        
        Model_used[id_model]={"name":"poly-{}".format(order),"family":"poly","order": order,"m theta": [0]*(order+1),"w theta": w_theta_poly_2[0:order+1]   ,"b bx": int(np.ceil(np.log2((order+1)*nb_max_bit_theta)))}   
        id_model+=1 
    
    
       
    
    #"""     
    #Predictives samples family
    for eta in range(0,1): 
        for order in range(1,3):
            Model_used[id_model]={"name":"samp.-{}-{}".format(order,eta),"family":"pred samples","model used":0, "order" : order,"eta" :eta , "m theta" : [0.]*order,"w theta" : [2]*order, "b bx": int(np.ceil(np.log2(order*nb_max_bit_theta)))}    
            id_model+=1    
    
    
    #Predictives parametric family
    for factor in [2]:#,5,10,50,100]:
        Model_used[id_model]={"name":"para.-{}".format(factor),"family":"pred para", "model used":0, "factor" : factor, "m theta" : [0.75,0*fn,0],"w theta":[0.5/factor,0.2/factor,2*np.pi/factor],"b bx": int(np.ceil(np.log2(3*nb_max_bit_theta_pred_para)))} 
        id_model+=1
    
    Model_used[id_model]={"name":"none","family":"none","b bx": 0}
    
    
        
    # bilan models used :
    for id_model in Model_used:
        print("id: {}, ".format(id_model),"name: {}".format(Model_used[id_model]["name"]))
        
    
           
    Residual_used = {}
    Residual_used[0]={"name":"DCT+BPC"}
    Residual_used[1]={"name":"DWT+BPC"}
    
    for id_residual in Residual_used:
        print("id: {}, ".format(id_residual),"name: {}".format(Residual_used[id_residual]["name"]))
        
    
    
    
        
    

    
    
    dict_MMC_enc={}
    dict_MMC_dec={}
    for k in range(nb_phase):
        dict_MMC_enc[k]=Encode_one_window(fn,fs,N,Model_used,Residual_used,verbose)
        dict_MMC_dec[k]=Decode_one_window(fn,fs,N,Model_used,Residual_used,verbose)
        
   
    
    x_rec=np.zeros((nb_phase,N*nb_w))
    
    
    SNR=np.zeros((nb_phase,nb_w))
    
    RMSE=np.zeros((nb_phase,nb_w))
    
    
    x_model=np.zeros((nb_phase,N*nb_w))
    
    x_residual=np.zeros((nb_phase,N*nb_w))
    
    
    
    SNR_m=np.zeros((nb_phase,nb_w))
    
    SNR_r=np.zeros((nb_phase,nb_w))
    
    R_m=np.zeros((nb_phase,nb_w))
    R_r=np.zeros((nb_phase,nb_w))
    
    
    R_h=np.zeros((nb_phase,nb_w))
    
    
    L=np.zeros((nb_phase,nb_w))
    
    
    x_p=np.zeros((nb_phase,N*3))
    
    x_n=np.zeros((nb_phase,N*nb_w))
    
    b_phase=np.ones((nb_phase,nb_w),dtype='int')#*bmax
    

    
    tps1 = time.perf_counter()
    width=6
    
    
    
    
    
    
    
    
    for phase in range(nb_phase):#(48,49): 
    
    
        for w in range(nb_w):
            #tt=np.linspace(0,(N-1)/fs,N)
            x_test=x[phase][w*N:(w+1)*N]
            #x_test=np.array([230000*np.sin(2*np.pi*50*tt[i]+np.pi/6)+np.random.normal(0,400) for i in range(N)])
            ##### coder    
            
            
            dict_MMC_enc[phase].MMC_enc(x_test,x_p[phase],metric,quality_target,bmax)
            SNRenc=get_snr(x[phase][w*N:(w+1)*N],dict_MMC_enc[phase].x_rec_enc)
            
            #print("enc",dict_MMC_enc[phase].x_rec_enc)
            
            #"""
            ##### decoder
            dict_MMC_dec[phase].MMC_dec(dict_MMC_enc[phase].code,x_p[phase],bmax)
            SNRdec=get_snr(x[phase][w*N:(w+1)*N], dict_MMC_dec[phase].x_rec_dec)
    
            if SNRenc!=SNRdec:
                print(SNRenc,SNRdec)
                print("ERREUR !!!!!!!!!!!!!!!!!!!!!!!!!!")
               
                
    
            #"""

            x_n[phase][w*N:(w+1)*N]=x[phase][w*N:(w+1)*N]*2**(-dict_MMC_enc[phase].kx_enc)
            
          
            x_model[phase][w*N:(w+1)*N]=dict_MMC_enc[phase].x_model_enc
            x_residual[phase][w*N:(w+1)*N]=dict_MMC_enc[phase].x_residual_enc
    
            x_rec[phase][w*N:(w+1)*N]=x_model[phase][w*N:(w+1)*N]+x_residual[phase][w*N:(w+1)*N]
            
            
            SNR[phase][w]=get_snr(x[phase][w*N:(w+1)*N],x_rec[phase][w*N:(w+1)*N])
           
            
            RMSE[phase][w]=get_rmse(x[phase][w*N:(w+1)*N],x_rec[phase][w*N:(w+1)*N]) 
    
            
            SNR_m[phase][w]=get_snr(x[phase][w*N:(w+1)*N],x_model[phase][w*N:(w+1)*N])
            SNR_r[phase][w]=get_snr(x[phase][w*N:(w+1)*N]-x_model[phase][w*N:(w+1)*N],x_residual[phase][w*N:(w+1)*N])
    
            R_m[phase][w]=dict_MMC_enc[phase].bx_enc+dict_MMC_enc[phase].b_bx_enc
            R_r[phase][w]=dict_MMC_enc[phase].br_enc
            R_h[phase][w]=1+dict_MMC_enc[phase].b_kx_enc+dict_MMC_enc[phase].b_kr_enc+dict_MMC_enc[phase].bm+dict_MMC_enc[phase].bl
            
            
            
            if False :#dict_MMC_enc[phase].best_Model_used[dict_MMC_enc[phase].id_model_enc]["family"]=="pred para":
                #print(dict_MMC_enc[phase].best_Model_used[dict_MMC_enc[phase].id_model_enc]["model used"])
                
                #print("phase",phase,dict_MMC_enc[phase].best_Model_used[dict_MMC_enc[phase].id_model_enc]["model used"])
                M[signal][phase][w]=dict_MMC_enc[phase].best_Model_used[dict_MMC_enc[phase].id_model_enc]["model used"]
                
                
                L_pred_para[signal][phase][w]=1
           
                
            else : 
                
                M[signal][phase][w]=dict_MMC_enc[phase].id_model_enc 
            
            L[phase][w]=dict_MMC_enc[phase].id_residual_enc
            
        
            #mise à jour de x_p
            
            if w==0:
                x_p[phase][2*N:3*N]=x_rec[phase][w*N:(w+1)*N]
            else :
                if w==1:
                    x_p[phase][N:3*N]=x_rec[phase][(w-1)*N:(w+1)*N]
                else :
                    x_p[phase][0:3*N]=x_rec[phase][(w-2)*N:(w+1)*N]
                    
                    
               
            
            print(f"w={w+1:3}, ph:{phase+1:1}",end='')
            print(f", bmax={len(dict_MMC_enc[phase].code):3}",end='')
            print(f", SNR={SNR[phase][w]:5.2f} dB",end='')
            print(f", RMSE={RMSE[phase][w]:6.2f} V",end='')
            print(f", m={dict_MMC_enc[phase].m_enc:10}, l={dict_MMC_enc[phase].l_enc:7}",end='')
            print(f", SNR_m={SNR_m[phase][w]:4.1f} dB, SNR_r={SNR_r[phase][w]:4.1f} dB",end='')
            print(f", bh=b_kx+b_kr+bm+bl+b_bx={dict_MMC_enc[phase].b_kx_enc:1}+{dict_MMC_enc[phase].b_kr_enc:1}+{dict_MMC_enc[phase].bm:1}+{dict_MMC_enc[phase].bl_enc:1}+{dict_MMC_enc[phase].b_bx_enc:1}={dict_MMC_enc[phase].b_kx_enc+dict_MMC_enc[phase].b_kr_enc+dict_MMC_enc[phase].bm+dict_MMC_enc[phase].bl_enc+dict_MMC_enc[phase].b_bx_enc:2} b",end='')
            print(f", bx={dict_MMC_enc[phase].bx_enc} b, br={dict_MMC_enc[phase].br_enc:4} b",end='') 
            print(f", kx={dict_MMC_enc[phase].kx_enc:3}, kr={dict_MMC_enc[phase].kr_enc:2}",end="")
            print('') 
    
          
        #print("sum RMSE transform= {:.0f}".format(MSE_transform[0][w]+MSE_transform[1][w]+MSE_transform[2][w]))
        #print("sum RMSE= {:.0f}".format(MSE[0][w]+MSE[1][w]+MSE[2][w]))
        if w in []:# [48]:#w==27 or w==44 or w==47  :
    
            for phase in range(nb_phase):
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(x[phase][w*N:(w+1)*N],lw=1,label='x_real{}'.format(phase+1))
                plt.plot(x_rec[phase][w*N:(w+1)*N],lw=1,label='SNR_real={:.1f} dB, RMSE_real={:.1f}'.format(SNR[phase][w],RMSE[phase][w]))
                plt.xlabel('ind')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Phase: {}, Window index {}".format(phase+1,w+1))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                
                
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(x[phase][w*N:(w+1)*N],lw=1,label='x_transform{}'.format(phase+1))
                plt.plot(x_model[phase][w*N:(w+1)*N],lw=1,label='m: {}, SNR_m={:.1f} dB, RMSE_m={:.1f}'.format(dict_MMC_enc[phase].m_enc,SNR_m[phase][w],get_rmse(x[phase][w*N:(w+1)*N], x_model[phase][w*N:(w+1)*N])))
                plt.xlabel('ind')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Phase: {}, Window index {}, bx={} bits".format(phase+1,w+1,dict_MMC_enc[phase].bx_enc))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
        
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(x[phase][w*N:(w+1)*N]-x_model[phase][w*N:(w+1)*N],lw=1,label='r_{}'.format(phase+1))
                plt.plot(x_residual[phase][w*N:(w+1)*N],lw=1,label='l: {}, SNR_r={:.1f} dB'.format(dict_MMC_enc[phase].l_enc,SNR_r[phase][w]))
                plt.xlabel('ind')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Phase: {}, Window index {}, br={} bits".format(phase+1,w+1,dict_MMC_enc[phase].br_enc))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()        
                
                
    
    

    
    tps2 = time.perf_counter()
    
    
    

       
    print("times to encode the {} windows: {:.2f} s".format(nb_w,tps2 - tps1))
    
    
    for phase in range(0):
        
        print("SNR_mean_{}={:.2f} dB".format(str(phase+1),np.mean(SNR[phase])))
        print("code_mean_{}={:.2f}".format(str(phase+1),np.mean(R_m[phase]+R_r[phase]+R_h[phase])))
        
        plt.figure(figsize=(width,4), dpi=100)
        for k in range(0,nb_w):
            if k%2==0:
                plt.plot([i for i in range(k*N,(k+1)*N)],x_n[phase][k*N:(k+1)*N],color="red",lw=1)
            else :
                plt.plot([i for i in range(k*N,(k+1)*N)],x_n[phase][k*N:(k+1)*N],color="blue",lw=1)
        plt.xlabel('ind sample')
        plt.ylabel('Scale Voltage x phase : {}'.format(phase+1))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        #### First and second stage   
    
        ####  reconstructed signal in the original domain
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(x[phase],lw=1,label='x, phase: {}'.format(phase+1))
        plt.plot(x_rec[phase],lw=1,label='x_rec, phase: {}, SNR={:.2f} dB, RMSE={:.2f} V'.format(phase+1,np.mean(SNR[phase]),np.mean(RMSE[phase])))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.title("Reconstruted signal in the original domain")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        """
        
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot((x[phase]-x_rec[phase]),lw=1,label='x-x_rec, phase: {}'.format(phase+1))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude')
        plt.title("Error of reconstruted signal in the original domain")
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        
        
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(SNR[phase],'-o',lw=1,label='SNR MMC {:.2f}, phase: {}'.format(np.mean(SNR[phase]),phase+1))
        plt.xlabel('ind window')
        plt.ylabel('SNR (dB)')
        plt.title("SNR obtain for each window in the original domain")
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        
        
        """
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(RMSE[phase],'-o',lw=1,label='RMSE MMC {:.0f}, phase: {}'.format(np.mean(RMSE[phase]),phase+1))
        plt.xlabel('ind window')
        plt.ylabel('RMSE')
        plt.title("RMSE obtain for each window in the original domain")
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        
    
        
    
        
        #### first stage
        
        """
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(x_model[phase]/1000,lw=1,label='x_model, phase: {}'.format(phase+1))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude x10e3')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        
        
        
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(SNR_m[phase],'-o',lw=1,label='SNR model MMC {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
        
        
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(R_m[phase],'-o',lw=1,label='Number of bits to code the model MMC {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('bm (bits)')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
        """
    
    
    
    
    
        #### Second stage
        
        """
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot((x[phase]-x_model[phase])/1000,lw=1,label='r_{}'.format(phase+1))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude x10e3')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
    
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(x_residual[phase]/1000,lw=1,label='r_rec_{}'.format(phase+1))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude x10e3')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot((x[phase]-x_model[phase]-x_residual[phase])/1000,lw=1,label='r_{}-r_rec_{}'.format(phase+1,phase+1))
        plt.xlabel('ind sample')
        plt.ylabel('Magnitude x10e3')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(SNR_r[phase],'-o',lw=1,label='SNR residual MMC {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
    
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(R_r[phase],'-o',lw=1,label='Number of bits to code the residual MMC {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('br (bits)')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
        
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(R_m[phase] + R_r[phase],'-o',lw=1,label='Number of bits to code the model + residual MMC {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('br (bits)')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()      
        """
        
        #for k in range(nb_w):
        #    if R_h[phase][k]+R_m[phase][k]+R_r[phase][k]>=800:
        #        R_r[phase][k]=850-(R_h[phase][k]+R_m[phase][k])
                
        
        largeur_barre = 0.5
        plt.figure(figsize=(width,4), dpi=100)
        plt.bar([k for k in range(nb_w)],R_m[phase], width = largeur_barre,color='r')
        plt.bar([k for k in range(nb_w)],R_r[phase], width = largeur_barre, bottom =R_m[phase],color='b')
        plt.bar([k for k in range(nb_w)],R_h[phase], width = largeur_barre, bottom =R_m[phase]+R_r[phase],color='g')
        plt.xlabel('ind window')
        plt.ylabel('bits')
        plt.legend(["nx", "nr", "nh"])
        plt.title("bits mean : {:.1f} bits".format(np.mean(R_m[phase]+R_r[phase]+R_h[phase])))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #print("bits_not_used={:.3f}".format(np.mean(R_not_used[phase])) )
        
        
        largeur_barre = 0.5
        plt.figure(figsize=(width,4), dpi=100)
        plt.bar([k for k in range(nb_w)],SNR_m[phase], width = largeur_barre,color='r')
        plt.bar([k for k in range(nb_w)],SNR_r[phase], width = largeur_barre, bottom = [np.max([0,SNR_m[phase][k]]) for k in range(nb_w)],color='b')
        plt.xlabel('ind window')
        plt.ylabel('SNR')
        plt.legend(["SNR transform model ", "SNR transform residual"])
        plt.title("SMR mean : {:.2f} dB".format(np.mean(SNR[phase])))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show() 
    
        
    

        yticks_labels = [Model_used[id_model]["name"] for id_model in Model_used]# if Model_used[id_model]["family"]!="pred para" ]
        yticks_positions = np.arange(len(yticks_labels))
        
        
        

        

        plt.figure(figsize=(width,6), dpi=100)
        plt.plot(0,0,color="red",label="pred. para.")
        for w in range(nb_w) :
            M[signal][phase][w]=np.min([yticks_positions[-1], M[signal][phase][w]])
            if L_pred_para[signal][phase][w]==1:
                plt.plot(w,M[signal][phase][w],'o',lw=1,color="red")
            else :
                plt.plot(w,M[signal][phase][w],'o',lw=1,color="b")     
        plt.xlabel('ind window')
        plt.ylabel('Model index')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.yticks(yticks_positions, yticks_labels)
        plt.show() 
        
        
    
    
        ### Transform used
        yticks_labels =  [Residual_used[id_residual]["name"] for id_residual in Residual_used]+["none"]
        yticks_positions = np.arange(len(yticks_labels))
        plt.figure(figsize=(width,4), dpi=100)
        plt.plot(L[phase],'o',lw=1,label='Transform used, phase: {}'.format(phase+1))
        plt.xlabel('ind window')
        plt.ylabel('Transform index')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.yticks(yticks_positions, yticks_labels)
        plt.show() 
        
    
#good

M_model_rate_200=[[[0., 6., 6., 6., 8., 0., 6., 8., 6., 6., 6., 6., 6., 0., 0., 8.,
        8., 7., 0., 6., 9., 0., 8., 7., 0., 8., 6., 0., 8., 8., 0., 8.,
        7., 6., 8., 8., 8., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 6., 8., 6., 8., 8., 7., 6., 7., 6., 6., 6., 6., 6., 8., 6.,
        6., 6., 6., 6., 6., 6., 6., 8., 6., 6., 7., 6., 6., 6., 6., 8.,
        6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 7.,
        8., 6., 6., 6.],
       [0., 8., 6., 6., 8., 0., 6., 6., 6., 6., 6., 8., 6., 0., 0., 0.,
        0., 8., 8., 8., 8., 8., 0., 7., 0., 8., 6., 0., 8., 8., 0., 8.,
        7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6.,
        6., 8., 6., 8., 8., 8., 8., 6., 8., 8., 7., 6., 8., 6., 6., 8.,
        6., 6., 8., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 7., 6., 6., 6., 8., 6., 6., 6., 6., 6., 7., 6., 6., 6.,
        6., 8., 6., 6.],
       [0., 6., 7., 6., 8., 0., 6., 6., 6., 6., 6., 6., 6., 0., 5., 0.,
        0., 7., 0., 6., 1., 0., 0., 8., 8., 8., 6., 0., 8., 8., 7., 7.,
        8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 6., 6., 7., 6., 6., 8., 6., 8., 7., 6., 8., 6., 6., 6.,
        7., 6., 6., 7., 6., 6., 6., 6., 8., 8., 6., 8., 8., 6., 7., 6.,
        8., 6., 6., 6., 7., 6., 6., 8., 6., 8., 7., 6., 8., 6., 6., 6.,
        6., 6., 8., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        8., 0., 4., 5., 3., 4., 8., 3., 9., 3., 4., 4., 1., 2., 8., 4.,
        5., 4., 0., 1., 2., 8., 2., 4., 4., 2., 8., 2., 9., 2., 1., 8.,
        8., 8., 9., 2., 6., 6., 6., 8., 2., 8., 6., 8., 0., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        8., 8., 0., 9., 4., 3., 4., 3., 9., 3., 5., 4., 2., 8., 3., 4.,
        5., 4., 9., 3., 1., 9., 2., 4., 9., 2., 1., 2., 4., 2., 1., 9.,
        1., 6., 2., 6., 8., 2., 6., 0., 6., 2., 6., 6., 6., 0., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 8., 6., 6., 6., 8., 6., 7., 6., 6., 6., 8., 6., 0.,
        0., 8., 5., 5., 9., 3., 4., 4., 9., 3., 5., 4., 2., 2., 3., 3.,
        5., 8., 9., 2., 6., 9., 2., 4., 3., 2., 1., 2., 4., 2., 1., 8.,
        9., 1., 6., 6., 1., 6., 9., 1., 6., 6., 6., 9., 9., 1., 6., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        6., 6., 1., 0., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 9., 0.,
        1., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 9., 0.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 0., 0., 0., 6., 6., 6., 6., 6., 6., 6., 0., 9.,
        3., 6., 8., 8., 8., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 0.,
        6., 5., 0., 6., 9., 5., 0., 8., 8., 0., 0., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 6., 8., 8., 6.,
        8., 8., 8., 8., 8., 8., 6., 6., 6., 6., 7., 6., 7., 6., 6., 6.,
        8., 8., 6., 6., 8., 8., 8., 8., 6., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8.],
       [0., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 0., 8.,
        6., 4., 0., 7., 0., 4., 0., 8., 8., 0., 8., 6., 7., 6., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8., 8.,
        6., 8., 8., 8., 8., 8., 7., 6., 6., 6., 7., 6., 8., 8., 8., 6.,
        7., 6., 8., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8.],
       [0., 8., 6., 8., 8., 8., 8., 8., 6., 6., 8., 8., 8., 8., 0., 0.,
        6., 0., 8., 6., 5., 9., 0., 8., 0., 8., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8.,
        6., 8., 8., 6., 8., 8., 8., 6., 7., 6., 8., 8., 6., 6., 7., 6.,
        8., 8., 6., 8., 6., 8., 8., 8., 6., 8., 8., 8., 8., 8., 6., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 8., 8.,
        6., 8., 8., 8.]],[[0., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 8., 0., 0., 8., 3.,
        0., 6., 8., 9., 2., 8., 9., 2., 2., 8., 2., 2., 8., 1., 8., 6.,
        2., 8., 6., 1., 1., 6., 6., 6., 8., 2., 1., 1., 9., 9., 1., 6.,
        6., 6., 6., 1., 9., 9., 9., 9., 9., 9., 9., 9., 6., 0., 8., 7.,
        8., 6., 8., 8., 8., 8., 8., 8., 0., 6., 8., 7., 6., 8., 8., 8.,
        8., 8., 8., 6., 8., 8., 7., 6., 7., 6., 8., 8., 6., 6., 6., 6.,
        8., 6., 6., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 0., 0., 0., 8., 4.,
        5., 0., 8., 4., 2., 8., 9., 2., 8., 3., 2., 8., 8., 1., 6., 8.,
        2., 6., 2., 2., 1., 6., 8., 6., 8., 2., 6., 1., 8., 1., 6., 1.,
        6., 6., 6., 6., 9., 9., 9., 9., 1., 9., 9., 9., 6., 0., 8., 7.,
        6., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8.,
        6., 0., 6., 6., 6., 8., 6., 8., 6., 6., 8., 7., 6., 6., 7., 6.,
        8., 6., 6., 6.],
       [0., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0., 8., 0., 4.,
        0., 0., 8., 9., 2., 2., 9., 2., 2., 4., 2., 8., 8., 1., 6., 6.,
        1., 2., 6., 1., 1., 6., 6., 6., 8., 2., 6., 6., 8., 1., 8., 6.,
        6., 6., 6., 6., 9., 9., 9., 1., 6., 6., 9., 9., 6., 9., 0., 7.,
        8., 6., 8., 8., 8., 7., 6., 8., 8., 8., 0., 6., 8., 8., 7., 6.,
        6., 6., 6., 8., 6., 8., 6., 6., 6., 6., 8., 6., 6., 6., 8., 6.,
        8., 6., 6., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        0., 9., 9., 5., 2., 9., 5., 2., 3., 3., 1., 2., 8., 1., 4., 5.,
        2., 5., 3., 5., 3., 2., 5., 5., 4., 8., 2., 3., 8., 5., 3., 2.,
        6., 1., 6., 2., 8., 3., 3., 1., 2., 6., 8., 2., 6., 1., 1., 6.,
        8., 6., 1., 6., 1., 6., 9., 9., 1., 6., 6., 6., 6., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 8., 3., 0.,
        8., 1., 9., 3., 8., 5., 9., 2., 3., 2., 8., 8., 2., 8., 4., 5.,
        2., 5., 3., 5., 3., 1., 5., 8., 3., 8., 3., 4., 3., 8., 2., 6.,
        8., 8., 8., 8., 6., 2., 1., 2., 2., 2., 8., 1., 6., 8., 6., 6.,
        8., 8., 1., 8., 8., 8., 8., 9., 1., 8., 6., 6., 6., 6., 6., 6.,
        1., 6., 1., 9., 9., 9., 9., 9., 1., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 5., 0.,
        8., 8., 3., 5., 3., 1., 5., 4., 3., 2., 4., 1., 8., 8., 4., 5.,
        2., 5., 2., 5., 3., 2., 5., 3., 4., 3., 2., 3., 8., 8., 2., 2.,
        2., 8., 1., 6., 8., 2., 1., 6., 1., 6., 8., 6., 6., 6., 2., 6.,
        2., 1., 6., 6., 6., 6., 8., 6., 8., 1., 6., 9., 9., 1., 8., 6.,
        6., 6., 6., 1., 9., 9., 1., 6., 6., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 8., 6., 6., 7., 6., 6., 6., 7., 8., 6., 6., 6., 0., 0.,
        0., 0., 0., 8., 6., 5., 5., 4., 5., 5., 5., 4., 4., 5., 4., 4.,
        8., 5., 5., 4., 8., 8., 8., 8., 8., 4., 4., 8., 8., 4., 4., 4.,
        8., 4., 8., 8., 8., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 6.,
        4., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 4., 4., 8.,
        8., 6., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 8., 8.,
        6., 6., 4., 4.],
       [0., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 8., 0.,
        0., 8., 8., 8., 7., 0., 5., 5., 5., 4., 8., 4., 5., 4., 5., 8.,
        5., 8., 5., 5., 8., 8., 8., 8., 8., 5., 8., 8., 8., 8., 8., 5.,
        5., 8., 8., 8., 5., 8., 8., 8., 5., 8., 8., 8., 8., 8., 8., 5.,
        8., 5., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5.,
        8., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5., 8.,
        8., 5., 8., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 9., 5.,
        0., 8., 8., 6., 8., 8., 9., 3., 4., 3., 4., 2., 8., 2., 4., 2.,
        3., 8., 2., 2., 2., 8., 1., 2., 6., 3., 2., 8., 1., 8., 2., 8.,
        8., 2., 2., 8., 8., 8., 1., 2., 8., 8., 2., 1., 8., 2., 6., 2.,
        8., 2., 2., 8., 8., 8., 6., 1., 1., 8., 1., 8., 6., 2., 6., 8.,
        2., 6., 0., 2., 8., 8., 6., 6., 6., 1., 6., 9., 1., 6., 2., 6.,
        6., 6., 8., 2.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0., 8.,
        0., 8., 7., 6., 0., 8., 6., 8., 8., 0., 6., 6., 8., 8., 5., 9.,
        3., 2., 8., 8., 2., 8., 1., 1., 2., 4., 5., 2., 8., 4., 3., 3.,
        1., 6., 8., 1., 9., 0., 7., 6., 8., 8., 8., 0., 6., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 6., 8., 6., 6., 6., 6.,
        6., 8., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 8., 6., 6., 6.,
        7., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0., 8.,
        8., 0., 7., 8., 0., 8., 6., 8., 8., 0., 7., 6., 8., 8., 9., 9.,
        3., 3., 9., 2., 1., 3., 2., 1., 6., 4., 5., 2., 8., 4., 3., 8.,
        1., 6., 2., 6., 9., 0., 6., 8., 7., 8., 8., 0., 6., 8., 8., 8.,
        8., 8., 8., 6., 7., 6., 8., 8., 8., 8., 8., 6., 8., 6., 6., 8.,
        7., 6., 8., 6., 6., 6., 6., 6., 7., 6., 8., 8., 8., 6., 6., 6.,
        8., 6., 7., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0., 0.,
        8., 8., 6., 8., 5., 0., 7., 6., 8., 0., 6., 8., 8., 8., 0., 9.,
        3., 2., 8., 8., 4., 8., 6., 1., 3., 4., 5., 3., 2., 4., 3., 3.,
        2., 6., 1., 8., 8., 0., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8., 6.,
        8., 6., 8., 8., 8., 8., 7., 6., 8., 7., 6., 8., 8., 6., 7., 6.,
        8., 6., 7., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8.,
        0., 0., 8., 0., 0., 0., 6., 8., 7., 8., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 7., 6., 6., 8., 7., 6., 8., 8., 7., 8., 6.,
        7., 7., 6., 8., 8., 8., 6., 7., 6., 7., 8., 6., 6., 7., 6., 7.,
        6., 6., 7., 6., 7., 6., 6., 6., 6., 7., 6., 6., 8., 6., 6., 6.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 7., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5.,
        8., 0., 5., 0., 8., 6., 8., 8., 8., 8., 7., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8., 6., 6., 7., 6., 8.,
        7., 6., 8., 8., 6., 8., 7., 6., 8., 6., 7., 6., 6., 7., 6., 6.,
        7., 6., 6., 7., 6., 7., 6., 7., 6., 6., 7., 6., 7., 6., 7., 6.,
        6., 8., 6., 7., 6., 7., 6., 6., 6., 6., 7., 6., 6., 8., 8., 6.,
        6., 6., 6., 8.],
       [0., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        8., 0., 9., 8., 0., 8., 7., 8., 6., 7., 8., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 7., 8., 6., 6., 8., 8., 8., 8., 8., 7., 6., 7.,
        6., 7., 6., 8., 8., 6., 7., 6., 8., 7., 6., 8., 6., 6., 6., 6.,
        6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 7., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 7., 6., 6.,
        6., 6., 8., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 8., 0., 6., 0., 6., 8., 4., 9.,
        5., 8., 9., 0., 8., 7., 6., 8., 7., 6., 7., 0., 8., 0., 0., 0.,
        0., 8., 7., 8., 8., 6., 7., 6., 8., 8., 8., 8., 8., 6., 8., 8.,
        8., 6., 8., 6., 8., 8., 6., 8., 8., 8., 8., 7., 0., 8., 6., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 7., 6., 8., 8., 8., 8., 6., 8., 6., 6., 7., 6., 6.,
        8., 8., 6., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 8., 0., 6., 0., 6., 6., 9., 5.,
        7., 8., 1., 0., 0., 8., 6., 7., 8., 6., 8., 7., 0., 8., 8., 0.,
        8., 0., 7., 8., 6., 8., 8., 7., 6., 6., 8., 8., 8., 7., 6., 7.,
        6., 7., 6., 8., 8., 8., 6., 7., 6., 8., 6., 7., 0., 8., 0., 6.,
        8., 8., 8., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 7., 6., 7., 6., 8., 8., 8., 8., 8., 6., 8., 8., 8., 6., 8.,
        8., 8., 6., 8.],
       [0., 6., 6., 6., 6., 8., 6., 6., 0., 8., 6., 0., 6., 6., 9., 1.,
        9., 8., 9., 0., 0., 8., 6., 8., 7., 6., 8., 7., 4., 0., 0., 0.,
        0., 8., 7., 8., 8., 6., 7., 6., 8., 7., 6., 8., 8., 8., 8., 7.,
        6., 8., 8., 8., 8., 6., 7., 6., 8., 8., 8., 7., 9., 0., 8., 6.,
        8., 7., 6., 8., 8., 7., 8., 6., 8., 7., 8., 8., 8., 6., 8., 7.,
        6., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 8., 6., 6., 7., 6.,
        8., 6., 6., 7.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        0., 0., 0., 8., 1., 9., 9., 8., 9., 9., 9., 9., 9., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        3., 5., 0., 2., 9., 1., 9., 9., 9., 9., 9., 9., 9., 8., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 8., 6., 6., 6., 8., 6., 7., 6., 6., 6.,
        5., 0., 0., 9., 2., 9., 1., 6., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        8., 9., 9., 1., 0., 8., 7., 6., 8., 7., 8., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 6., 8., 6., 8., 6., 8., 8., 6., 6.,
        8., 6., 6., 8., 6., 6., 8., 7., 6., 6., 8., 8., 8., 6., 8., 6.,
        8., 8., 8., 6., 6., 8., 6., 6., 6., 8., 6., 8., 6., 6., 8., 6.,
        8., 8., 6., 6.],
       [0., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        8., 8., 8., 8., 8., 8., 7., 6., 8., 7., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 7., 6., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8.,
        8., 8., 6., 8., 6., 8., 6., 8., 7., 6., 8., 6., 8., 7., 6., 6.,
        8., 8., 8., 6., 8., 7., 6., 6., 8., 7., 6., 6., 8., 8., 8., 6.,
        7., 6., 6., 6., 8., 8., 6., 7., 6., 8., 6., 8., 6., 7., 6., 8.,
        6., 8., 6., 8.],
       [0., 6., 6., 6., 6., 7., 6., 7., 6., 7., 6., 6., 6., 6., 5., 0.,
        8., 9., 9., 0., 0., 0., 7., 6., 8., 7., 8., 6., 8., 8., 8., 8.,
        8., 8., 8., 7., 6., 6., 8., 8., 8., 8., 8., 6., 8., 8., 8., 7.,
        6., 8., 8., 8., 8., 8., 6., 8., 7., 6., 6., 6., 7., 6., 6., 8.,
        6., 6., 8., 6., 8., 6., 6., 6., 7., 6., 8., 6., 7., 6., 8., 6.,
        8., 6., 8., 8., 8., 8., 6., 8., 6., 6., 6., 7., 6., 6., 6., 7.,
        6., 8., 6., 6.]]]


"""
#sans gamma1
M_model_rate_200=[[[0., 6., 6., 6., 8., 0., 6., 6., 7., 6., 6., 7., 6., 9., 0., 8.,
        8., 7., 0., 6., 9., 0., 0., 7., 0., 8., 6., 0., 0., 8., 8., 8.,
        7., 6., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6.,
        8., 8., 8., 8., 8., 7., 8., 6., 8., 8., 6., 6., 8., 6., 6., 8.,
        8., 8., 6., 6., 6., 7., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 8., 6., 6., 8., 0., 6., 8., 6., 6., 6., 6., 6., 8., 0., 8.,
        0., 7., 0., 6., 0., 8., 8., 7., 0., 8., 6., 0., 8., 8., 0., 0.,
        7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8.,
        8., 6., 8., 8., 7., 6., 8., 6., 8., 7., 6., 6., 6., 8., 6., 8.,
        6., 6., 8., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 8.,
        7., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6.,
        6., 8., 6., 6.],
       [0., 6., 7., 6., 8., 0., 6., 6., 6., 6., 6., 6., 6., 8., 5., 0.,
        0., 7., 0., 6., 1., 0., 0., 7., 0., 8., 6., 0., 8., 8., 7., 7.,
        7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 6., 8., 7., 6., 7., 6., 7., 8., 6., 8., 6., 6., 8., 7., 8.,
        6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 8., 6., 6., 8., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 7., 6., 7.,
        6., 8., 6., 6.]],[[0., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 7., 6., 6., 6., 0.,
        8., 0., 4., 5., 3., 4., 8., 3., 9., 3., 4., 4., 1., 2., 8., 4.,
        3., 4., 9., 1., 2., 8., 2., 4., 9., 2., 8., 2., 9., 2., 1., 8.,
        8., 8., 9., 2., 6., 6., 6., 8., 2., 8., 6., 8., 0., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 0.,
        8., 8., 6., 9., 4., 8., 4., 3., 9., 3., 5., 4., 2., 8., 3., 4.,
        5., 4., 9., 3., 1., 9., 2., 4., 9., 2., 1., 2., 9., 2., 1., 9.,
        1., 6., 2., 6., 8., 2., 6., 0., 6., 2., 6., 6., 6., 0., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 7., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 6., 0.,
        0., 7., 5., 9., 9., 3., 4., 4., 9., 3., 5., 4., 2., 2., 3., 3.,
        5., 8., 9., 2., 6., 9., 2., 4., 3., 2., 1., 2., 8., 2., 1., 8.,
        9., 1., 6., 6., 1., 6., 9., 1., 6., 6., 6., 9., 9., 1., 6., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        6., 6., 9., 0., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 9., 0.,
        1., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 9., 0.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 7., 0., 0., 0., 6., 6., 6., 6., 6., 6., 6., 0., 8.,
        3., 6., 8., 8., 8., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 0.,
        6., 8., 0., 6., 9., 5., 0., 0., 8., 0., 0., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7.,
        6., 6., 8., 8., 8., 6., 8., 8., 6., 8., 7., 6., 6., 8., 8., 6.,
        8., 8., 6., 8., 8., 7., 6., 8., 8., 8., 8., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8.],
       [0., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 0., 8.,
        6., 4., 0., 6., 0., 4., 0., 8., 8., 8., 8., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 8., 8., 8., 8.,
        8., 8., 6., 8., 8., 8., 8., 6., 8., 8., 8., 7., 6., 8., 8., 6.,
        8., 8., 7., 6., 8., 8., 8., 7., 6., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8.],
       [0., 8., 6., 8., 8., 8., 8., 8., 6., 6., 8., 8., 8., 8., 8., 0.,
        6., 0., 8., 6., 9., 9., 0., 8., 8., 8., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 6., 6., 8., 8., 6., 6., 8., 7., 6., 6., 6., 6., 6.,
        6., 8., 7., 6., 8., 8., 8., 8., 6., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0., 8., 8., 2.,
        0., 6., 8., 9., 2., 8., 9., 2., 2., 8., 2., 2., 8., 1., 8., 6.,
        2., 8., 6., 1., 1., 6., 6., 6., 8., 2., 1., 1., 9., 9., 1., 6.,
        6., 6., 6., 1., 9., 9., 9., 9., 9., 9., 9., 9., 6., 0., 8., 7.,
        8., 6., 8., 8., 8., 6., 8., 8., 0., 6., 8., 8., 8., 8., 8., 8.,
        7., 6., 8., 6., 6., 8., 8., 8., 7., 6., 8., 7., 6., 6., 8., 8.,
        8., 6., 6., 7.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0., 8., 8., 4.,
        5., 0., 8., 4., 2., 8., 9., 2., 8., 3., 2., 8., 8., 1., 6., 8.,
        2., 6., 2., 2., 1., 6., 8., 6., 8., 2., 6., 1., 8., 1., 6., 1.,
        6., 6., 6., 6., 9., 9., 9., 9., 1., 9., 9., 9., 6., 0., 8., 7.,
        6., 8., 7., 6., 8., 7., 6., 8., 8., 8., 8., 0., 6., 6., 8., 6.,
        8., 8., 8., 8., 6., 8., 7., 6., 6., 6., 8., 6., 6., 6., 6., 6.,
        8., 6., 7., 6.],
       [0., 7., 6., 6., 6., 6., 6., 6., 8., 6., 8., 8., 0., 8., 0., 4.,
        0., 0., 8., 9., 2., 2., 9., 2., 2., 4., 2., 8., 8., 1., 6., 6.,
        1., 2., 6., 1., 1., 6., 6., 6., 8., 2., 6., 6., 8., 1., 8., 6.,
        6., 6., 6., 6., 9., 9., 9., 1., 6., 6., 9., 9., 6., 9., 0., 7.,
        6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 6.,
        6., 0., 6., 6., 6., 6., 6., 8., 6., 7., 6., 6., 6., 6., 8., 6.,
        6., 6., 6., 8.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        8., 9., 9., 5., 2., 9., 5., 2., 3., 3., 1., 2., 8., 1., 4., 5.,
        2., 5., 3., 5., 3., 2., 5., 5., 4., 8., 2., 3., 8., 5., 3., 2.,
        6., 1., 6., 2., 8., 3., 3., 1., 2., 6., 8., 2., 6., 1., 1., 6.,
        8., 6., 1., 6., 1., 6., 9., 9., 1., 6., 6., 6., 6., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 8., 6., 6., 6., 7., 6., 7., 6., 6., 6., 8., 3., 0.,
        8., 1., 9., 3., 8., 8., 9., 2., 3., 2., 8., 8., 2., 8., 4., 5.,
        2., 5., 3., 5., 3., 1., 5., 8., 3., 8., 3., 4., 3., 8., 2., 6.,
        8., 8., 8., 8., 6., 2., 1., 2., 2., 2., 8., 1., 6., 8., 6., 6.,
        8., 8., 1., 8., 8., 8., 8., 9., 1., 8., 6., 6., 6., 6., 6., 6.,
        1., 6., 1., 9., 9., 9., 9., 9., 1., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.],
       [0., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 8., 0.,
        8., 6., 3., 5., 3., 1., 5., 4., 3., 2., 4., 1., 8., 8., 4., 5.,
        2., 5., 2., 5., 3., 2., 5., 3., 4., 3., 2., 3., 8., 8., 2., 2.,
        2., 8., 1., 6., 8., 2., 1., 6., 1., 6., 8., 6., 6., 6., 2., 6.,
        2., 1., 6., 6., 6., 6., 8., 6., 8., 1., 6., 9., 9., 1., 8., 6.,
        6., 6., 6., 1., 9., 9., 1., 6., 6., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 9., 0.,
        8., 8., 8., 0., 6., 5., 5., 4., 5., 5., 5., 4., 4., 5., 4., 4.,
        8., 5., 5., 4., 8., 8., 8., 8., 8., 4., 4., 8., 8., 4., 4., 4.,
        8., 4., 8., 8., 8., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 6.,
        4., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 4., 4., 8.,
        8., 6., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 8., 8.,
        6., 6., 4., 4.],
       [0., 6., 6., 7., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 8., 0.,
        8., 8., 8., 8., 6., 0., 4., 5., 5., 4., 8., 4., 5., 4., 5., 8.,
        5., 8., 5., 5., 8., 8., 8., 8., 8., 5., 8., 8., 8., 8., 8., 5.,
        5., 8., 8., 8., 5., 8., 8., 8., 5., 8., 8., 8., 8., 8., 8., 5.,
        8., 5., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5.,
        8., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5., 8.,
        8., 5., 8., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 9., 5.,
        0., 0., 6., 8., 8., 9., 9., 3., 4., 3., 4., 2., 8., 2., 4., 2.,
        3., 8., 2., 2., 2., 8., 1., 2., 6., 3., 2., 8., 1., 8., 2., 8.,
        8., 2., 2., 8., 8., 8., 1., 2., 8., 8., 2., 1., 8., 2., 6., 2.,
        8., 2., 2., 8., 8., 8., 6., 1., 1., 8., 1., 8., 6., 2., 6., 8.,
        2., 6., 0., 2., 8., 8., 6., 6., 6., 1., 6., 9., 1., 6., 2., 6.,
        6., 6., 8., 2.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0.,
        8., 8., 6., 8., 0., 8., 6., 8., 8., 0., 6., 7., 6., 8., 9., 9.,
        3., 2., 8., 8., 2., 8., 1., 1., 2., 4., 5., 2., 8., 4., 3., 3.,
        1., 6., 8., 1., 9., 0., 7., 6., 8., 8., 8., 0., 6., 8., 8., 8.,
        8., 7., 6., 8., 8., 8., 8., 8., 6., 6., 6., 6., 6., 8., 6., 7.,
        6., 8., 6., 6., 6., 6., 7., 6., 6., 8., 8., 6., 6., 8., 6., 8.,
        7., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0.,
        8., 8., 7., 8., 0., 8., 6., 8., 8., 0., 6., 6., 8., 8., 9., 9.,
        3., 3., 9., 1., 8., 3., 2., 1., 6., 4., 5., 2., 8., 4., 3., 8.,
        1., 6., 2., 6., 9., 0., 6., 8., 8., 8., 8., 0., 6., 6., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 6., 8., 8., 8., 8., 6., 8., 8., 7.,
        6., 6., 8., 8., 8., 8., 6., 8., 6., 6., 6., 6., 7., 8., 6., 6.,
        8., 7., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 0.,
        8., 8., 6., 8., 5., 0., 6., 8., 8., 0., 6., 6., 8., 8., 8., 9.,
        3., 2., 8., 8., 4., 8., 6., 1., 3., 4., 5., 3., 2., 4., 3., 3.,
        2., 6., 1., 8., 8., 0., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 7., 6., 8., 6., 7., 6., 8., 8., 6., 6., 7., 6., 8., 6.,
        6., 8., 6., 7.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 6., 8.,
        0., 8., 8., 0., 8., 8., 6., 8., 8., 8., 7., 6., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 7., 6., 8., 8., 8., 7., 6., 8., 8., 8., 7.,
        8., 8., 6., 6., 8., 8., 7., 6., 8., 7., 8., 6., 8., 8., 6., 6.,
        6., 6., 7., 6., 8., 6., 6., 6., 6., 7., 8., 6., 7., 6., 6., 6.,
        6., 6., 6., 8., 6., 7., 6., 6., 6., 6., 7., 6., 8., 7., 6., 6.,
        6., 6., 8., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5.,
        8., 0., 5., 0., 8., 6., 8., 8., 8., 8., 7., 6., 8., 8., 7., 6.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8.,
        7., 8., 6., 8., 6., 6., 7., 6., 7., 8., 8., 6., 6., 6., 6., 6.,
        6., 8., 7., 6., 6., 7., 6., 7., 6., 7., 8., 7., 8., 8., 6., 6.,
        6., 6., 8., 8., 6., 7., 6., 6., 6., 6., 7., 8., 6., 7., 6., 6.,
        6., 6., 6., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 0.,
        8., 0., 9., 8., 0., 0., 7., 8., 8., 8., 8., 8., 8., 8., 6., 8.,
        8., 8., 8., 8., 7., 6., 8., 6., 7., 6., 7., 6., 7., 6., 6., 8.,
        6., 8., 7., 8., 6., 6., 7., 6., 8., 7., 6., 7., 8., 6., 6., 8.,
        8., 6., 7., 6., 6., 7., 6., 8., 6., 7., 6., 6., 6., 7., 6., 7.,
        6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 7., 6., 7., 6., 6.,
        6., 6., 6., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 8., 0., 6., 0., 6., 8., 4., 9.,
        5., 8., 9., 0., 8., 7., 6., 8., 7., 8., 8., 6., 0., 8., 0., 8.,
        8., 8., 7., 8., 6., 8., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 6., 8., 8., 8., 8., 7., 8., 0., 6., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 6., 6., 8., 6., 6.,
        8., 6., 8., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 8., 0., 6., 0., 6., 6., 9., 5.,
        7., 8., 9., 0., 8., 8., 6., 8., 7., 6., 7., 6., 0., 0., 8., 0.,
        8., 0., 7., 8., 6., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8., 8.,
        7., 6., 8., 6., 7., 6., 8., 6., 8., 8., 6., 7., 6., 0., 8., 6.,
        7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8., 8., 8.,
        8., 8., 6., 8.],
       [0., 6., 6., 6., 6., 7., 6., 6., 8., 0., 6., 0., 6., 6., 9., 1.,
        9., 8., 9., 0., 8., 0., 6., 8., 8., 8., 8., 7., 4., 0., 0., 8.,
        8., 0., 7., 8., 6., 8., 8., 8., 8., 8., 8., 7., 6., 8., 6., 8.,
        8., 8., 8., 6., 8., 6., 8., 6., 8., 8., 8., 7., 9., 0., 6., 8.,
        8., 7., 6., 8., 8., 7., 8., 6., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 7., 6., 8., 6., 8., 7.,
        6., 6., 6., 8.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        0., 8., 0., 8., 1., 9., 9., 8., 9., 9., 9., 9., 9., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        3., 5., 0., 2., 9., 1., 9., 9., 9., 9., 9., 9., 9., 8., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        5., 0., 0., 9., 2., 9., 1., 6., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 7., 7., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        7., 9., 9., 1., 0., 0., 7., 6., 8., 7., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7., 6., 8.,
        8., 8., 8., 8., 7., 6., 7., 8., 6., 6., 8., 7., 6., 8., 6., 7.,
        6., 6., 8., 8., 8., 7., 6., 8., 6., 6., 8., 6., 8., 6., 8., 6.,
        8., 6., 8., 6., 6., 8., 6., 6., 6., 6., 6., 6., 8., 6., 6., 8.,
        6., 6., 6., 6.],
       [0., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        8., 8., 8., 8., 8., 8., 7., 6., 8., 7., 6., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6., 8., 7., 6., 6., 8.,
        8., 8., 8., 8., 7., 6., 6., 6., 6., 8., 8., 6., 8., 8., 6., 8.,
        8., 8., 6., 7., 6., 6., 8., 6., 8., 8., 7., 6., 8., 8., 7., 6.,
        8., 6., 8., 8., 8., 8., 8., 6., 8., 8., 6., 8., 6., 6., 6., 6.,
        8., 8., 6., 6.],
       [0., 6., 7., 6., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 8., 0.,
        8., 9., 9., 8., 0., 0., 7., 6., 8., 7., 8., 6., 7., 6., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6.,
        8., 8., 8., 8., 7., 6., 6., 8., 6., 8., 8., 6., 7., 6., 6., 8.,
        6., 8., 6., 8., 8., 6., 7., 6., 8., 6., 8., 8., 6., 7., 6., 6.,
        8., 6., 8., 6., 8., 6., 6., 8., 6., 6., 8., 8., 6., 8., 8., 6.,
        6., 8., 6., 8.]]]

"""
M_BF_200=[[[0., 6., 6., 6., 6., 0., 6., 6., 7., 6., 6., 7., 6., 8., 9., 0.,
        0., 6., 0., 6., 9., 9., 0., 9., 7., 0., 6., 9., 0., 8., 7., 0.,
        7., 6., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 8., 6., 6., 6., 8., 6., 6., 6., 8., 6., 6., 6.,
        6., 8., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 7., 9., 6., 6., 6., 6., 6., 6., 6., 6., 0., 8.,
        8., 7., 6., 8., 9., 0., 0., 7., 8., 9., 6., 9., 9., 0., 7., 8.,
        7., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 8., 6., 6., 7., 6., 6., 7., 6., 6., 7., 6., 7., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6.,
        6., 6., 6., 8.],
       [0., 6., 7., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 9., 9., 0.,
        9., 6., 9., 9., 9., 9., 0., 7., 7., 0., 6., 9., 9., 9., 7., 7.,
        7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 8., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 8.,
        6., 8., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6.,
        6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 6., 6.,
        8., 6., 6., 6.]],[[0., 6., 7., 6., 6., 6., 7., 6., 6., 6., 6., 7., 6., 6., 6., 8.,
        0., 7., 9., 3., 4., 4., 8., 3., 2., 3., 4., 4., 1., 2., 8., 4.,
        2., 4., 3., 1., 2., 3., 2., 4., 4., 2., 8., 2., 2., 2., 1., 8.,
        8., 8., 9., 2., 6., 6., 6., 8., 2., 8., 6., 8., 0., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9.,
        0., 6., 9., 2., 4., 3., 4., 3., 2., 3., 5., 4., 2., 8., 3., 4.,
        5., 4., 2., 3., 1., 3., 2., 4., 8., 2., 1., 2., 2., 2., 1., 9.,
        1., 6., 2., 6., 8., 2., 6., 0., 6., 2., 6., 6., 6., 0., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 5.,
        1., 7., 5., 5., 5., 3., 4., 4., 5., 3., 5., 4., 2., 2., 3., 3.,
        3., 4., 2., 2., 6., 3., 2., 4., 4., 2., 1., 2., 4., 2., 1., 8.,
        9., 1., 6., 6., 1., 6., 9., 1., 6., 6., 6., 9., 9., 1., 6., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5., 0.,
        6., 6., 4., 0., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 9., 9.,
        1., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 0., 0., 0., 7., 6., 6., 6., 6., 6., 6., 2., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 0., 0., 0., 6., 6., 6., 6., 6., 6., 6., 9., 9.,
        3., 6., 8., 8., 8., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9.,
        8., 9., 9., 6., 8., 9., 3., 9., 9., 0., 7., 6., 7., 6., 6., 6.,
        6., 6., 8., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6.,
        6., 6., 6., 8., 8., 6., 8., 6., 6., 7., 6., 8., 6., 8., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [9., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 0.,
        6., 0., 8., 6., 3., 4., 9., 9., 9., 9., 7., 8., 8., 6., 6., 6.,
        6., 6., 6., 6., 7., 6., 7., 6., 6., 7., 6., 6., 6., 8., 6., 8.,
        6., 6., 6., 6., 6., 6., 8., 6., 7., 6., 6., 6., 6., 6., 6., 7.,
        6., 6., 6., 6., 6., 6., 6., 7., 8., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 7., 6., 6., 8., 9.,
        6., 0., 8., 6., 4., 8., 8., 9., 0., 8., 7., 6., 7., 6., 6., 6.,
        6., 6., 6., 6., 7., 6., 6., 8., 6., 7., 6., 6., 6., 6., 6., 8.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 7., 6., 8., 6., 6., 8., 6.,
        6., 6., 6., 7., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 7.,
        6., 6., 6., 6.]],[[0., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 9., 9., 9., 2.,
        0., 9., 8., 9., 2., 8., 2., 2., 2., 8., 2., 2., 8., 1., 8., 6.,
        2., 8., 6., 1., 1., 6., 6., 6., 8., 2., 1., 1., 9., 9., 1., 6.,
        6., 6., 6., 1., 9., 9., 9., 9., 9., 9., 9., 9., 2., 0., 8., 6.,
        6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 8., 6., 6., 8., 6., 7., 6., 6., 6., 6., 6., 6., 8., 6., 6.,
        6., 7., 6., 6.],
       [0., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 8., 9., 0., 0., 3.,
        0., 6., 8., 4., 2., 8., 2., 2., 8., 3., 2., 8., 8., 1., 6., 8.,
        2., 6., 2., 2., 1., 6., 8., 6., 8., 2., 6., 1., 8., 1., 6., 1.,
        6., 6., 6., 6., 9., 9., 9., 9., 1., 9., 9., 9., 3., 0., 8., 6.,
        8., 7., 6., 6., 7., 8., 6., 8., 8., 8., 6., 6., 8., 6., 6., 8.,
        6., 6., 8., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 7.],
       [0., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 8., 0., 0., 0., 9.,
        9., 8., 6., 9., 2., 2., 2., 8., 8., 4., 2., 8., 8., 1., 6., 6.,
        1., 2., 6., 1., 1., 6., 6., 6., 8., 2., 6., 6., 8., 1., 8., 6.,
        6., 6., 6., 6., 9., 9., 9., 1., 6., 6., 9., 9., 9., 0., 8., 6.,
        6., 7., 6., 7., 6., 7., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 8., 6., 6., 6., 7., 6., 6., 6., 8., 6., 6., 8., 6.,
        6., 6., 6., 7.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 0.,
        7., 6., 4., 5., 2., 9., 2., 2., 3., 3., 1., 2., 8., 1., 4., 5.,
        8., 9., 3., 5., 3., 2., 5., 2., 4., 8., 2., 3., 8., 5., 3., 2.,
        6., 1., 6., 2., 8., 3., 3., 1., 2., 6., 8., 2., 6., 1., 1., 6.,
        8., 6., 1., 6., 1., 6., 9., 9., 1., 6., 6., 6., 6., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 2., 9.,
        0., 4., 2., 3., 8., 2., 5., 2., 3., 2., 8., 8., 2., 8., 4., 5.,
        2., 9., 3., 9., 3., 1., 5., 3., 8., 8., 8., 4., 3., 8., 2., 6.,
        8., 8., 8., 8., 6., 2., 1., 2., 2., 2., 8., 1., 6., 8., 6., 6.,
        8., 8., 1., 8., 8., 8., 8., 9., 1., 8., 6., 6., 6., 6., 6., 6.,
        1., 6., 1., 9., 9., 9., 9., 9., 1., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 4., 0.,
        7., 8., 2., 5., 3., 2., 5., 4., 3., 2., 4., 1., 8., 8., 4., 5.,
        2., 9., 2., 5., 3., 2., 5., 3., 4., 3., 2., 3., 8., 8., 2., 2.,
        2., 8., 1., 6., 8., 2., 1., 6., 1., 6., 8., 6., 6., 6., 2., 6.,
        2., 1., 6., 6., 6., 6., 8., 6., 8., 1., 6., 9., 9., 1., 8., 6.,
        6., 6., 6., 1., 9., 9., 1., 6., 6., 6., 6., 6., 6., 6., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 7., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 8., 0.,
        0., 8., 8., 7., 6., 5., 2., 3., 5., 5., 9., 4., 4., 5., 4., 4.,
        8., 5., 5., 4., 8., 8., 8., 8., 8., 4., 4., 8., 8., 4., 4., 4.,
        8., 4., 8., 8., 8., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 6.,
        4., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 4., 4., 8.,
        8., 6., 4., 4., 8., 8., 8., 8., 8., 8., 8., 4., 8., 8., 8., 8.,
        6., 6., 4., 4.],
       [0., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 0.,
        8., 8., 0., 6., 7., 8., 9., 2., 9., 4., 8., 4., 5., 4., 5., 8.,
        5., 6., 5., 5., 8., 8., 8., 8., 8., 5., 8., 8., 8., 8., 8., 5.,
        5., 8., 8., 8., 5., 8., 8., 8., 5., 8., 8., 8., 8., 8., 8., 5.,
        8., 5., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5.,
        8., 8., 8., 8., 8., 8., 8., 5., 8., 8., 6., 5., 8., 8., 5., 8.,
        8., 5., 8., 8.],
       [0., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 5., 8.,
        0., 6., 6., 7., 6., 8., 9., 2., 4., 3., 4., 2., 8., 2., 4., 2.,
        3., 8., 2., 2., 2., 8., 1., 2., 6., 3., 2., 8., 1., 8., 2., 8.,
        8., 2., 2., 8., 8., 8., 1., 2., 8., 8., 2., 1., 8., 2., 6., 2.,
        8., 2., 2., 8., 8., 8., 6., 1., 1., 8., 1., 8., 6., 2., 6., 8.,
        2., 6., 0., 2., 8., 8., 6., 6., 6., 1., 6., 9., 1., 6., 2., 6.,
        6., 6., 8., 2.]],[[9., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9.,
        9., 0., 7., 6., 8., 0., 6., 6., 8., 0., 6., 6., 6., 6., 6., 9.,
        3., 2., 2., 2., 2., 8., 1., 1., 2., 4., 5., 2., 8., 9., 3., 3.,
        1., 6., 8., 1., 9., 9., 6., 7., 8., 6., 6., 6., 8., 6., 6., 6.,
        6., 6., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 6.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6.,
        6., 6., 7., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 9.,
        9., 0., 7., 8., 4., 0., 6., 6., 8., 9., 7., 6., 6., 6., 9., 9.,
        3., 3., 2., 3., 1., 3., 2., 1., 6., 4., 5., 2., 8., 2., 3., 8.,
        1., 6., 2., 6., 9., 0., 6., 7., 6., 6., 6., 9., 6., 7., 8., 6.,
        6., 6., 7., 6., 7., 6., 6., 7., 6., 6., 6., 8., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 8., 6., 6.,
        6., 8., 7., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9.,
        9., 0., 6., 7., 9., 9., 6., 8., 6., 9., 6., 8., 6., 6., 6., 9.,
        3., 2., 3., 3., 4., 8., 6., 1., 3., 4., 5., 3., 2., 2., 3., 3.,
        2., 6., 1., 8., 2., 9., 6., 7., 8., 8., 6., 6., 6., 6., 8., 6.,
        6., 8., 6., 6., 6., 6., 6., 8., 7., 6., 6., 6., 6., 6., 6., 6.,
        6., 7., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 6., 6.,
        6., 8., 7., 6.]],[[9., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 8.,
        9., 0., 9., 8., 6., 7., 6., 8., 7., 6., 6., 7., 8., 8., 6., 8.,
        6., 8., 7., 6., 7., 8., 8., 6., 7., 8., 6., 6., 6., 7., 6., 7.,
        6., 7., 6., 7., 6., 6., 7., 6., 8., 6., 7., 6., 6., 6., 6., 6.,
        6., 6., 7., 6., 6., 7., 6., 6., 7., 6., 7., 6., 7., 6., 6., 6.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 7., 8.,
        6., 6., 6., 8.],
       [9., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 5.,
        8., 0., 5., 0., 8., 6., 6., 6., 8., 6., 7., 6., 6., 7., 6., 6.,
        6., 6., 6., 6., 7., 8., 6., 8., 8., 6., 6., 6., 6., 7., 8., 6.,
        6., 7., 8., 6., 6., 6., 7., 6., 6., 7., 6., 7., 6., 7., 8., 6.,
        6., 7., 6., 6., 6., 6., 6., 6., 6., 7., 6., 7., 6., 7., 6., 6.,
        6., 6., 8., 7., 6., 7., 6., 6., 6., 6., 6., 7., 8., 8., 8., 6.,
        7., 6., 8., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        0., 8., 5., 9., 0., 6., 7., 7., 6., 6., 7., 8., 8., 6., 8., 6.,
        6., 6., 6., 6., 7., 8., 8., 8., 6., 7., 6., 6., 6., 7., 6., 7.,
        8., 6., 7., 6., 6., 7., 6., 6., 8., 7., 6., 6., 6., 6., 8., 6.,
        6., 6., 6., 7., 6., 7., 6., 6., 7., 6., 7., 6., 6., 7., 6., 6.,
        6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 8., 7., 6., 7., 8., 6.,
        7., 8., 6., 6.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 0., 8., 0., 7., 6., 3., 9.,
        9., 0., 5., 9., 6., 7., 7., 7., 8., 8., 8., 6., 5., 0., 8., 8.,
        8., 7., 7., 8., 8., 6., 7., 6., 6., 8., 6., 6., 6., 6., 8., 7.,
        6., 8., 8., 6., 6., 8., 6., 8., 6., 8., 6., 7., 6., 9., 6., 6.,
        7., 6., 8., 6., 6., 6., 6., 6., 6., 6., 6., 8., 6., 8., 8., 6.,
        6., 6., 6., 8., 6., 6., 8., 6., 6., 6., 8., 6., 7., 6., 6., 8.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 0., 7., 0., 6., 6., 4., 9.,
        7., 8., 3., 9., 7., 9., 6., 8., 6., 8., 8., 6., 9., 9., 0., 9.,
        8., 7., 7., 8., 6., 6., 7., 8., 8., 8., 8., 8., 8., 6., 8., 8.,
        8., 8., 8., 6., 8., 6., 6., 8., 6., 8., 6., 7., 0., 9., 7., 8.,
        6., 8., 8., 8., 6., 6., 8., 7., 6., 8., 8., 8., 8., 7., 8., 6.,
        8., 8., 8., 7., 8., 8., 6., 6., 8., 8., 6., 8., 6., 8., 8., 6.,
        8., 6., 6., 8.],
       [0., 7., 6., 6., 6., 7., 6., 6., 6., 0., 8., 8., 6., 6., 5., 0.,
        6., 7., 9., 9., 7., 9., 6., 6., 8., 8., 8., 7., 0., 0., 8., 8.,
        8., 8., 7., 8., 6., 6., 6., 8., 7., 8., 6., 8., 8., 8., 8., 6.,
        6., 8., 7., 6., 8., 8., 6., 7., 6., 8., 6., 7., 0., 8., 6., 7.,
        8., 6., 6., 6., 6., 6., 6., 8., 8., 8., 7., 6., 6., 8., 6., 8.,
        7., 6., 6., 6., 6., 7., 6., 7., 8., 6., 8., 6., 8., 7., 6., 6.,
        6., 6., 6., 8.]],[[0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        4., 0., 7., 9., 1., 9., 9., 8., 9., 9., 9., 9., 9., 1., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        3., 0., 6., 3., 9., 1., 9., 9., 9., 9., 9., 9., 9., 8., 9., 9.,
        9., 9., 9., 9., 9., 9., 2., 0., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.],
       [0., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        5., 0., 0., 4., 2., 9., 1., 6., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 2., 2., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
        9., 9., 9., 9.]],[[0., 6., 6., 6., 8., 7., 6., 7., 6., 7., 6., 6., 6., 6., 4., 0.,
        6., 6., 9., 0., 9., 9., 7., 8., 6., 7., 6., 6., 6., 7., 6., 8.,
        6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 8., 6.,
        6., 8., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 7., 6., 6.,
        6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 7., 8., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 7., 6., 6., 6., 6., 6., 8., 6.,
        6., 8., 6., 6.],
       [9., 6., 6., 6., 8., 7., 6., 6., 6., 6., 6., 6., 6., 6., 2., 0.,
        8., 9., 9., 9., 6., 9., 6., 8., 6., 7., 8., 6., 7., 6., 6., 6.,
        6., 6., 6., 7., 6., 7., 6., 6., 6., 6., 8., 8., 7., 6., 8., 6.,
        6., 8., 6., 7., 8., 6., 6., 6., 6., 6., 6., 8., 7., 6., 6., 6.,
        6., 6., 6., 6., 8., 6., 6., 8., 6., 6., 6., 7., 6., 6., 6., 8.,
        6., 6., 6., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 6.],
       [0., 6., 6., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 0.,
        6., 9., 0., 8., 0., 0., 7., 6., 8., 6., 7., 6., 6., 6., 7., 6.,
        6., 8., 6., 6., 6., 6., 6., 7., 6., 6., 7., 6., 6., 6., 6., 6.,
        7., 6., 6., 6., 7., 6., 6., 6., 6., 6., 7., 6., 6., 7., 6., 6.,
        6., 7., 6., 6., 7., 6., 6., 6., 6., 6., 8., 6., 6., 6., 6., 6.,
        7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        8., 7., 6., 6.]]]

                                            
                                                                
Matrice_confusion=np.zeros((10,10))
nb_w=100
for signal in range(12):
    for phase in range(3):
        for w in range(nb_w):
          
            i=int(np.min([9,M_BF_200[signal][phase][w]]))
            j=int(np.min([9,M_model_rate_200[signal][phase][w]]))
            Matrice_confusion[i,j]+=1
    
print(Matrice_confusion)       
                                                                   
# Récupération des dimensions de la matrice de confusion
num_classes = Matrice_confusion.shape[0]

row_sums = Matrice_confusion.sum(axis=1)
#print(row_sums)
Matrice_confusion = Matrice_confusion / row_sums[:, np.newaxis] #


# Création du graphique
plt.figure(figsize=(10, 8))
plt.imshow(Matrice_confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()

# Définition des étiquettes des axes
tick_marks = np.arange(num_classes)

name_model=['sin-1','poly-0', 'poly-1', 'poly-2', 'poly-3', 'poly-4', 'samp.-1', 'samp.-2','para', 'none']
plt.xticks(tick_marks, name_model, rotation=45)
plt.yticks(tick_marks, name_model)

# Affichage des valeurs dans les cellules de la matrice de confusion
thresh = Matrice_confusion.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(100*Matrice_confusion[i, j], '.0f'),
                 ha="center", va="center",
                 color="white" if Matrice_confusion[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('BF model')
plt.xlabel('Predict model')
plt.show()    

for i in range(num_classes):
    S=np.sum(Matrice_confusion[i])
    for j in range(num_classes):
        print(j,i,"{:.0f}".format(100*Matrice_confusion[i,j]/S ))       
    print("")                                                 
                                                                