# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: presvôts
"""

from MMC_test_model import Encode_one_window,Decode_one_window
from Measures import get_snr,get_rmse
import numpy as np
import matplotlib.pyplot as plt

#%%
"""
# Open the data
"""

Delta_u=18.31055
Delta_i=4.314
loaded = np.load('DATA_S2.npz')    
DATA_S=loaded['DATA_S2']


for number in range(len(DATA_S)):
    
    DATA_S[number][0]*=Delta_u
    DATA_S[number][1]*=Delta_u
    DATA_S[number][2]*=Delta_u
    DATA_S[number][3]*=Delta_i
    DATA_S[number][4]*=Delta_i
    DATA_S[number][5]*=Delta_i

print("shape database",np.shape(DATA_S))

#%%
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



# bilan models used:
print("Recap set of models")
for id_model in Model_used:
    print("id model: {}, ".format(id_model),"name: {}".format(Model_used[id_model]["name"]))
    

"""
# Set of method of residual compression
"""   
    
Residual_used = {}

Residual_used[0]={"name":"none"}
Residual_used[1]={"name":"DCT+BPC"}
Residual_used[2]={"name":"DWT+BPC"}

# bilan models used:
print("Recap set of methods of residual compression")
for id_residual in Residual_used:
    print("id residual: {}, ".format(id_residual),"name: {}".format(Residual_used[id_residual]["name"]))
    

"""
# Initialisation MMC
"""  
    
#initialisation inputs

id_signal=6 #fault event from database
nb_w=100 # number of window per signals
nb_min=0 # start window in the signal

v1=DATA_S[id_signal][0]
v2=DATA_S[id_signal][1]
v3=DATA_S[id_signal][2]
i1=DATA_S[id_signal][3]
i2=DATA_S[id_signal][4]
i3=DATA_S[id_signal][5]


x=[v1[nb_min*N:N*(nb_min+nb_w)],v2[nb_min*N:N*(nb_min+nb_w)],v3[nb_min*N:N*(nb_min+nb_w)]] # creat input x of MMC
nb_phase=1#len(x) # number of phase encode 

#################################### visualisation of signal
t=np.linspace(0,(nb_w)*(N-1)*(1/fs),nb_w*N)

plt.figure(figsize=(8,4), dpi=100)
plt.plot(t,v1[nb_min*N:(nb_min+nb_w)*N]/1000,lw=1,label="v1")
plt.plot(t,v2[nb_min*N:(nb_min+nb_w)*N]/1000,lw=1,label="v2")
plt.plot(t,v3[nb_min*N:(nb_min+nb_w)*N]/1000,lw=1,label="v3")
plt.xlabel('t (s)')
plt.ylabel('voltage (kV)')
plt.title('Voltages from database, id signal: {:.0f}'.format(id_signal))
plt.legend()
plt.grid(which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

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

id_model=1
id_residual=1
n_max=128
n_x=30

for phase in range(nb_phase):
    for w in range(nb_w): 
        
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
        SNRenc=get_snr(x[phase][w*N:(w+1)*N],dict_MMC_enc[phase].x_rec_enc)
        
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

  
"""
# Visualisation of outputs 
"""  
for phase in range(nb_phase):
    width=8
    ####  reconstructed signal 
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,x[phase]/1000,lw=1,label='x')
    plt.plot(t,x_rec[phase]/1000,lw=1,label="x_rec")
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.title("Reconstruted signal phase: {}, SNR mean={:.2f} dB, RMSE mean={:.2f} V".format(phase+1,np.mean(SNR[phase]),np.mean(RMSE[phase])))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,(x[phase]-x_rec[phase]),lw=1,label='x-x_rec')
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude')
    plt.title("Error of reconstruted signal, phase: {}".format(phase+1))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],RMSE[phase],'-o',lw=1,label='RMSE')
    plt.xlabel('t (s)')
    plt.ylabel('RMSE (V)')
    plt.title('RMSE obtain for each window, RMSE mean={:.0f} V, phase: {}'.format(np.mean(RMSE[phase]),phase+1))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    #### first stage
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,x[phase]/1000,lw=1,label='x')
    plt.plot(t,x_model[phase]/1000,lw=1,label='x_model')
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.title('Reconstructed models, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,(x[phase]-x_model[phase])/1000,lw=1,label='x-x_model')
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.title('Error of reconstructed models, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],SNR_m[phase],'-o',lw=1,label='SNR models')
    plt.xlabel('t (s)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.title('SNR models, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],R_m[phase],'-o',lw=1,label='R_m')
    plt.xlabel('t (s)')
    plt.ylabel('rate (bits)')
    plt.legend()
    plt.title('Number of bits to code the model per window, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
  
    
    #### Second stage
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,(x[phase]-x_model[phase])/1000,lw=1,label='x-x_model')
    plt.plot(t,x_residual[phase]/1000,lw=1,label='x_residual')
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.title('Reconstructed residuals, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 

    plt.figure(figsize=(width,4), dpi=100)
    plt.plot(t,(x[phase]-x_model[phase]-x_residual[phase])/1000,lw=1,label='x-x_model-x_residual')
    plt.xlabel('t (s)')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.title('Error of reconstructed residual, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 

    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],SNR_r[phase],'-o',lw=1,label='SNR residual')
    plt.xlabel('t (s)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.title('Number of bits to code the residual per window, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  

    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],R_r[phase],'-o',lw=1,label='R_r')
    plt.xlabel('t (s)')
    plt.ylabel('rate (bits)')
    plt.legend()
    plt.title('Number of bits to code the residual, phase: {}'.format(phase+1))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()  
    
     
    # Contribution of two stages 
    largeur_barre = (2*N/3)/fs
    """
    plt.figure(figsize=(width,4), dpi=100)
    plt.bar([t[k] for k in range(0,nb_w*N,N)],100*R_h[phase]/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]), width = largeur_barre,color='g')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],100*R_m[phase]/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]), width = largeur_barre, bottom =100*R_h[phase]/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]),color='r')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],100*R_r[phase]/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]), width = largeur_barre, bottom =100*(R_h[phase]+R_m[phase])/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]),color='b')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],100*R_unused[phase]/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]), width = largeur_barre, bottom =100*(R_h[phase]+R_m[phase]+R_r[phase])/(R_unused[phase]+R_h[phase]+R_m[phase]+R_r[phase]),color='c')
    plt.xlabel('t (s)')
    plt.ylabel('Rate (%)')
    plt.legend(["R_h", "R_m", "R_r","R_unused"])
    plt.title("bits mean: {:.2f} bits, phase: {}".format(np.mean(R_m[phase]+R_r[phase]+R_h[phase]),phase))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    """
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.bar([t[k] for k in range(0,nb_w*N,N)],R_m[phase], width = largeur_barre,color='r')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],R_r[phase], width = largeur_barre, bottom =R_m[phase] ,color='b')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],R_h[phase], width = largeur_barre,bottom =R_m[phase]+R_r[phase],color='g')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],R_unused[phase], width = largeur_barre, bottom =R_h[phase]+R_m[phase]+R_r[phase],color='c')
    plt.xlabel('t (s)')
    plt.ylabel('Rate (%)')
    plt.legend(["R_h", "R_m", "R_r","R_unused"])
    plt.title("bits mean: {:.2f} bits, phase: {}".format(np.mean(R_m[phase]+R_r[phase]+R_h[phase]),phase))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(width,4), dpi=100)
    plt.bar([t[k] for k in range(0,nb_w*N,N)],SNR_m[phase], width = largeur_barre,color='r')
    plt.bar([t[k] for k in range(0,nb_w*N,N)],SNR_r[phase], width = largeur_barre, bottom = [np.max([0,SNR_m[phase][k]]) for k in range(nb_w)],color='b')
    plt.xlabel('t (s)')
    plt.ylabel('SNR (dB)')
    plt.legend(["SNR model ", "SNR residual"])
    plt.title("SMR mean : {:.2f} dB".format(np.mean(SNR[phase])))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    #Model used
    yticks_labels = [Model_used[id_model]["name"] for id_model in Model_used]
    yticks_positions = np.arange(len(yticks_labels))
    plt.figure(figsize=(width,7), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],M[phase],'o',lw=1,label='index model')
    plt.xlabel('t (s)')
    plt.ylabel('Model index')
    plt.legend()
    plt.title("index of best selected model, phase: {}".format(phase))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.yticks(yticks_positions, yticks_labels)
    plt.show() 
    
    # Method of residual compression used
    yticks_labels =  [Residual_used[id_residual]["name"] for id_residual in Residual_used]
    yticks_positions = np.arange(len(yticks_labels))
    plt.figure(figsize=(width,4), dpi=100)
    plt.plot([t[k] for k in range(0,nb_w*N,N)],L[phase],'o',lw=1,label='index method')
    plt.xlabel('t (s)')
    plt.ylabel('Index of method of residual compression')
    plt.legend()
    plt.title("Best selected method, phase: {}".format(phase))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.yticks(yticks_positions, yticks_labels)
    plt.show() 
    
