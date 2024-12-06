# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:37:29 2023

@author: presvotscor
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:41:57 2023

@author: presvotscor
"""


import numpy as np
import matplotlib.pyplot as plt


from Normalize import normalize


from codage_model import Model_Encoder,Model_Decoder
from codage_residu import Residual_Encoder,Residual_Decoder


#from Models import Model_sin,Model_poly
from Measures import get_rmse,get_snr,get_quality,my_bin,my_inv_bin
import copy
#from Bits_allocation import Allocation_sin,Allocation_poly,Allocation_pred_samples

class Encode_one_window(Model_Encoder,Residual_Encoder):#,Allocation_sin,Allocation_poly,Allocation_pred_samples):
    def __init__(self,fn=50,fs=6400, N=128, Model_used={},Residual_used={}, verbose=False):
        

        self.Model_used=Model_used # dictionnaire des modèles utilisés ainsi que leurs caractéristiques respéctives
        self.Residual_used=Residual_used 
      
        Model_Encoder.__init__(self,fn,fs,N,False) 
        Residual_Encoder.__init__(self,N)   

        
    
        
        ##################### budget de bits servant à décoder le signal

        
        self.bm=int(np.ceil(np.log2(len(Model_used)))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin ++samples+para+none
        
        self.bl=1  # nombre de bits pour coder la méthode de résidu actuelement: DCT, DWT
       
        self.b_kx=5 # nombre de bits pour coder kx, 0 si modèle pred_samples
        
        self.b_kr=4 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        
        self.stop_dicho=1
        
        
        
        
        
        """
        ################################ Allocation optimal de bits 
        Allocation_sin.__init__(self,N,1/fs,max_bits) 
        Allocation_poly.__init__(self,N,max_bits) 
        Allocation_pred_samples.__init__(self,N,max_bits) 
        
        
        for id_model in self.Model_used:
            if self.best_Model_used[id_model]["family"]=="sin":
                self.best_Model_used[id_model]["allocation"]=self.get_all_nx_sin(,w_theta_poly,dtype='int')
                
        nb_poly_max=10
        vect_allocation_poly=[]
        for order in range()
        vect_allocation_poly=self.get_all_nx_poly(,w_theta_poly,dtype='int')
        """
    def ini_MMC_enc(self,metric,quality,bmax):
        self.metric=metric
        self.quality=quality
        self.bmax=bmax
        
        self.best_Model_used=copy.deepcopy(self.Model_used)
        self.best_Residual_used=copy.deepcopy(self.Residual_used)
        
        
        

    
    def verif_order(self,A, B, C):

        # Vérification de la condition A[i] < B[i] < C[i] pour chaque élément
        for i in range(len(A)):
            if not (A[i]-0.5*C[i] <= B[i] <= A[i]+0.5*C[i]):
                return False
    
        return True


    def get_case(self,x_n,x_p_n):
        
        #print("theta_sin_hat",theta_sin_hat)
        
        bad_model=[]
        del_model=[]
        for name in self.Model_used:
            if self.best_Model_used[name]["family"]=="sin":
                theta_sin_hat=self.get_theta_sin(x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                self.best_Model_used[name]["theta hat"]=theta_sin_hat
                
                if self.verif_order(self.best_Model_used[name]["m theta"],theta_sin_hat,self.best_Model_used[name]["w theta"]):
                    
                    
                    # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                    self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                    self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_sin_tilde(theta_sin_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                    self.best_Model_used[name]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[name]["theta tilde"])
                    self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                    if self.best_Model_used[name]["quality model"]>self.quality_n:
                        bad_model.append(name)
                        
                else :
                    del_model.append(name)
                    #del self.best_Model_used[name] 
                    
                    
            elif self.best_Model_used[name]["family"]=="poly":
                theta_poly_hat=self.get_theta_poly(x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"],self.best_Model_used[name]["order"])
                self.best_Model_used[name]["theta hat"]=theta_poly_hat
                
                """
                print("--------------------")
                for k in range(self.best_Model_used[name]["order"]+1):
                    print(-self.best_Model_used[name]["w theta"][k]/2,theta_poly_hat[k],self.best_Model_used[name]["w theta"][k]/2)
                """
                
                if self.verif_order(self.best_Model_used[name]["m theta"],theta_poly_hat,self.best_Model_used[name]["w theta"]):
                   
                    
                    # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                    self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                    self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_poly_tilde(theta_poly_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                    self.best_Model_used[name]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[name]["theta tilde"])
                    self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                    if self.best_Model_used[name]["quality model"]>self.quality_n:
                        bad_model.append(name)
                else :
                    del_model.append(name)
                    #del self.best_Model_used[name]
                    
            elif self.best_Model_used[name]["family"]=="pred samples":

                if self.best_Model_used[name]["model used"]!="pred samples":
                    m_theta_pred_samples=self.get_m_theta_pred_samples(self.best_Model_used[name]["order"],self.best_Model_used[name]["eta"],0,[0]*self.best_Model_used[name]["order"],[np.infty]*self.best_Model_used[name]["order"]) 
                else :   
                    X_pred_samples2=self.get_X(x_p_n[0:2*self.N],self.best_Model_used[name]["order"],self.best_Model_used[name]["eta"]) 
                    m_theta_pred_samples=self.get_theta_pred_samples(X_pred_samples2,x_p_n[2*self.N:3*self.N],[0]*self.best_Model_used[name]["order"],[10]*self.best_Model_used[name]["order"]) 
                #print("m_theta_pred_samples enc",m_theta_pred_samples)
                self.best_Model_used[name]["m theta"]=m_theta_pred_samples
                
                
                
                X_pred_samples=self.get_X(x_p_n[self.N:3*self.N],self.best_Model_used[name]["order"],self.best_Model_used[name]["eta"])
                theta_pred_samples_hat=self.get_theta_pred_samples(X_pred_samples,x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                
                
                #vérification de si le modèle est bon
                self.best_Model_used[name]["X"]=X_pred_samples
                self.best_Model_used[name]["theta hat"]=theta_pred_samples_hat

                #print("m theta",self.best_Model_used[name]["m theta"])
                if self.verif_order(m_theta_pred_samples,theta_pred_samples_hat,self.best_Model_used[name]["w theta"]):
                    
                    # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                    self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                    self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_pred_samples_tilde(theta_pred_samples_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                    self.best_Model_used[name]["x model"]=self.get_model_pred_samples(X_pred_samples,*self.best_Model_used[name]["theta tilde"])
                    self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                    
                    

                    
                    if self.best_Model_used[name]["quality model"]>self.quality_n:
                        bad_model.append(name)
                else :
                    del_model.append(name)
                    #del self.best_Model_used[name]
                    
                    
                    
                    
            elif self.best_Model_used[name]["family"]=="pred para":
                name_model_used=self.best_Model_used[name]["model used"]

                
                if self.Model_used[name_model_used]["family"]=="sin":
                    
                    theta_sin_hat=self.get_theta_sin(x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                    
                    self.best_Model_used[name]["theta hat"]=theta_sin_hat
                    """                       
                    self.best_Model_used[name]["m theta"]=self.Model_used[name_model_used]["theta tilde"]
                    factor=self.best_Model_used[name]["factor"]
                    self.best_Model_used[name]["w theta"]=[self.Model_used[name_model_used]["w theta"][i]/factor for i in range(3)]
                    """
                     
                    
                    if self.verif_order(self.best_Model_used[name]["m theta"],theta_sin_hat,self.best_Model_used[name]["w theta"]):

                        # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                        self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                        self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_sin_tilde(theta_sin_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                        self.best_Model_used[name]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[name]["theta tilde"])
                        self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                        if self.best_Model_used[name]["quality model"]>self.quality_n:
                            bad_model.append(name)
                    else :
                        del_model.append(name)
                        #del self.best_Model_used[name]
                elif self.Model_used[name_model_used]["family"]=="poly":
                    theta_poly_hat=self.get_theta_poly(x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"],self.best_Model_used[name_model_used]["order"])
                    self.best_Model_used[name]["theta hat"]=theta_poly_hat
                    
                    """
                    self.best_Model_used[name]["m theta"]=self.Model_used[name_model_used]["theta tilde"]
                    factor=self.best_Model_used[name]["factor"]
                    self.best_Model_used[name]["w theta"]=[self.Model_used[name_model_used]["w theta"][i]/factor for i in range(self.Model_used[name_model_used]["order"]+1)]
                    """
                        
                    if self.verif_order(self.best_Model_used[name]["m theta"],theta_poly_hat,self.best_Model_used[name]["w theta"]):

                        # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                        self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                        

                        self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_poly_tilde(theta_poly_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                        self.best_Model_used[name]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[name]["theta tilde"])
                        self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)                        
                        if self.best_Model_used[name]["quality model"]>self.quality_n:
                            bad_model.append(name)    

                    else :
                        del_model.append(name)
                        #del self.best_Model_used[name]
                elif self.Model_used[name_model_used]["family"]=="pred samples":
                    
                    #print("aaaaaaaaaaaaaaaaaaaaaaaaaa")

                    #print("self.best_Model_used[name_model_used]",self.best_Model_used[name_model_used])
                    #theta_pred_samples_hat=self.best_Model_used[name_model_used]["theta hat"]
                    
                    X_pred_samples=self.get_X(x_p_n[self.N:3*self.N],self.best_Model_used[name_model_used]["order"],self.best_Model_used[name_model_used]["eta"])
                    theta_pred_samples_hat=self.get_theta_pred_samples(X_pred_samples,x_n,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                    
                    """
                    self.best_Model_used[name]["m theta"]= self.Model_used[name_model_used]["theta tilde"]
                    
                    factor=self.best_Model_used[name]["factor"]
                    self.best_Model_used[name]["w theta"]=[self.Model_used[name_model_used]["w theta"][i]/factor for i in range(self.Model_used[name_model_used]["order"])]
                    """
                    
                    
                    if self.verif_order(self.best_Model_used[name]["m theta"],theta_pred_samples_hat,self.best_Model_used[name]["w theta"]):
                        self.best_Model_used[name]["theta hat"]=theta_pred_samples_hat
                        
                        #print(self.best_Model_used[name])
                        # Determination du modèle recostruit en utilisant le budget de bits maximal autorisé
                        self.best_Model_used[name]["X"]=self.best_Model_used[name_model_used]["X"]
                        self.best_Model_used[name]["bx"]=2**self.best_Model_used[name]["b bx"]-1
                        
                        
  
                        
                        
                        
                        
                        self.best_Model_used[name]["theta tilde"],self.best_Model_used[name]["code"]=self.get_theta_pred_samples_tilde(theta_pred_samples_hat,self.best_Model_used[name]["bx"],self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                        self.best_Model_used[name]["x model"]=self.get_model_pred_samples(self.best_Model_used[name]["X"],*self.best_Model_used[name]["theta tilde"])
                        self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                        if self.best_Model_used[name]["quality model"]>self.quality_n:
                            bad_model.append(name)
                    else :
                        del_model.append(name)
                        #del self.best_Model_used[name]
            
            
            elif self.best_Model_used[name]["family"]=="none":
                self.best_Model_used[name]["bx"]=0
                self.best_Model_used[name]["code"]=[]
                self.best_Model_used[name]["x model"]=np.zeros(self.N)
                self.best_Model_used[name]["quality model"]=get_quality(x_n,self.best_Model_used[name]["x model"],self.metric)
                if self.best_Model_used[name]["quality model"]>self.quality_n:
                    bad_model.append(name)
            else:     
                print("le modèle {} n'existe pas".format(name))
        
        for name in del_model:
            del self.best_Model_used[name]
        print("model to delete",del_model)   
        #jgcgcgc
        if len(bad_model)!=len(self.best_Model_used): #il y a au moins un bon modèle
            for name in bad_model:
                del self.best_Model_used[name]
            return True
        return False
                    
                    
            
                    

                    
    
    
    def enc_model(self,x,name,bx):
        
     
   
        
        if self.best_Model_used[name]["family"]=="pred samples":
            
            
            theta_tilde_test,code_theta_tilde_test=self.get_theta_pred_samples_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
            
            x_rec_test=self.get_model_pred_samples(self.best_Model_used[name]["X"],*theta_tilde_test) 
            quality_test=get_quality(x,x_rec_test,self.metric)

         
        elif self.best_Model_used[name]["family"]=="pred para":

            id_previous_model=self.best_Model_used[name]["model used"]
            if self.Model_used[id_previous_model]["family"]=="sin":
                theta_tilde_test,code_theta_tilde_test=self.get_theta_sin_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                x_rec_test=self.get_model_sin(self.t,*theta_tilde_test) 
                quality_test=get_quality(x,x_rec_test,self.metric)
                
            elif self.Model_used[ id_previous_model]["family"]=="pred samples":
                theta_tilde_test,code_theta_tilde_test=self.get_theta_pred_samples_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
             
                
                
                X=self.best_Model_used[name]["X"]
                #print("X",X)
                x_rec_test=self.get_model_pred_samples(X,*theta_tilde_test) 
                quality_test=get_quality(x,x_rec_test,self.metric)
               
            elif self.Model_used[id_previous_model]["family"]=="poly":
                theta_tilde_test,code_theta_tilde_test=self.get_theta_poly_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
                x_rec_test=self.get_model_poly(self.t,*theta_tilde_test) 
                quality_test=get_quality(x,x_rec_test,self.metric)
                  
        elif self.best_Model_used[name]["family"]=="sin":
            
     

            theta_tilde_test,code_theta_tilde_test=self.get_theta_sin_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
            x_rec_test=self.get_model_sin(self.t,*theta_tilde_test) 
            quality_test=get_quality(x,x_rec_test,self.metric)
             
    
        elif self.best_Model_used[name]["family"]=="poly":
            
    

            theta_tilde_test,code_theta_tilde_test=self.get_theta_poly_tilde(self.best_Model_used[name]["theta hat"],bx,self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"])
            x_rec_test=self.get_model_poly(self.t,*theta_tilde_test) 
            quality_test=get_quality(x,x_rec_test,self.metric)
        
        elif self.best_Model_used[name]["family"]=="none": 
            
            
            theta_tilde_test=[]
            code_theta_tilde_test=[]
            x_rec_test=np.zeros(self.N)
            quality_test=get_quality(x,x_rec_test,self.metric)  
        else :
            print("le modèle {} n'existe pas".format(name))
       

        return theta_tilde_test,code_theta_tilde_test,x_rec_test,quality_test 
        
    def enc_residual(self,r,quality_r):
        r_rec=[]
        code_r=[]
        id_r="none"
        for id_residual in self.best_Residual_used:
            #print("id_residual ",id_residual )
            #print("self.best_Residual_used",self.best_Residual_used)
            if self.best_Residual_used[id_residual]["name"]=="DCT+BPC":
                
                r_rec_test,code_r_test=self.get_r_DCT_BPC_tilde(r,self.metric,quality_r,self.bmax)
                #print("SNR_DCT",get_snr(r, r_rec_test))
                if len(r_rec)==0 or len(code_r_test)<len(code_r):
                    id_r=id_residual
                    r_rec=copy.copy(r_rec_test)
                    code_r=copy.copy(code_r_test)
                #br=len(code_r_test) 
            elif self.best_Residual_used[id_residual]["name"]=="DWT+BPC":
                r_rec_test,code_r_test=self.get_r_DWT_BPC_tilde(r,self.metric,quality_r,self.bmax)
                #print("SNR_DWT",get_snr(r, r_rec_test))
                if len(r_rec)==0 or len(code_r_test)<len(code_r):
                    id_r=id_residual
                    r_rec=copy.copy(r_rec_test)
                    code_r=copy.copy(code_r_test)
                #br=len(code_r_test) 
            else :
                return "la méthode est fausse"
                
        return r_rec,code_r,id_r
    
    
    def enc_residual2(self,r,quality_r,id_residual):
        r_rec=[]
        code_r=[]
        #id_r="none"

        if self.best_Residual_used[id_residual]["name"]=="DCT+BPC":
            
            r_rec_test,code_r_test=self.get_r_DCT_BPC_tilde(r,self.metric,quality_r,self.bmax)
            #print("SNR_DCT",get_snr(r, r_rec_test))
            if len(r_rec)==0 or len(code_r_test)<len(code_r):
                #id_r=id_residual
                r_rec=copy.copy(r_rec_test)
                code_r=copy.copy(code_r_test)
            #br=len(code_r_test) 
        elif self.best_Residual_used[id_residual]["name"]=="DWT+BPC":
            r_rec_test,code_r_test=self.get_r_DWT_BPC_tilde(r,self.metric,quality_r,self.bmax)
            #print("SNR_DWT",get_snr(r, r_rec_test))
            if len(r_rec)==0 or len(code_r_test)<len(code_r):
                #id_r=id_residual
                r_rec=copy.copy(r_rec_test)
                code_r=copy.copy(code_r_test)
            #br=len(code_r_test) 
        else :
            return "la méthode est fausse"
                
        return r_rec,code_r
        
    def MMC_enc(self,x,x_p,metric,quality,bmax):
        
        
        self.ini_MMC_enc(metric,quality,bmax)

        ##################### normalisation de x
        
        _,self.kx_enc=normalize(x)

        if self.kx_enc>=2**self.b_kx:
            self.kx_enc=2**self.b_kx-1
        
        if self.kx_enc<0:
            self.kx_enc==0
            
        x_n=copy.copy(x)*2**(-self.kx_enc)
        x_p_n=copy.copy(x_p)*2**(-self.kx_enc)
        
        if metric== "RMSE" or  metric=="MSE" :
            self.quality_n=quality*2**(-self.kx_enc)
        else :
            self.quality_n=quality
        
        
        
        
        ##################### détermination de theta_hat modèles pour les modèles conssidérés:  sin, poly 0,...,6
        cas_one=self.get_case(x_n,x_p_n)
        
        
        """
        print("cas one",cas_one)


        for name in self.best_Model_used:
            print("id : {}".format(name),"name: {},".format(self.best_Model_used[name]["name"]),"RMSE = {:.0f} V,".format(get_rmse(x_n,self.best_Model_used[name]["x model"])*2**self.kx_enc),"SNR = {:.0f} V,".format(get_snr(x_n,self.best_Model_used[name]["x model"])),"b model: {}".format(self.best_Model_used[name]["bx"]+self.best_Model_used[name]["b bx"]))
        """    

    
        ################ visualisation des modèles reconstruit

        """
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x_n,lw=1,label='x')
        for name in self.best_Model_used:
            
            plt.plot(self.best_Model_used[name]["x model"],lw=1,label='name={}, RMSE={:.0f} V, SNR={:.0f} dB'.format(name,get_rmse(x_n,self.best_Model_used[name]["x model"])*2**self.kx_enc,get_snr(x_n,self.best_Model_used[name]["x model"])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        """
        

        
        
        #### Cas 1  On a trouvé un sous ensemble de \mathcal{M}_{0} noté \mathcal{M}=\left\{ m_{1},\dots,m_{K}\right\} , K\geqslant1 tel que C\left(\boldsymbol{x},\boldsymbol{x}^{m_{k}}\left(\widehat{\boldsymbol{\theta}}\right)\right)\leqslant q_{\text{min}} avec m_{k}\in\mathcal{M}. (Pas besoins d'allouer des bits pour coder le résidu) 
        
        if cas_one:

            
            Imin=0   
            key=next(iter(self.best_Model_used))
            Imax=self.best_Model_used[key]["bx"]+self.best_Model_used[key]["b bx"]
            
            
            b_model=int(np.floor(Imax/2))
            bsup=Imax
            while Imax-Imin>self.stop_dicho:
        

                memory_bad_model=[]
                for name in self.best_Model_used : # on parcours tous les modèles en partant du meilleurs en qualité
                    bx=b_model-self.best_Model_used[name]["b bx"]
                    
                    
                    if (bx>=0) :#or (bx>=0 and self.best_Model_used[name]["name"]=="none"):

                        theta_tilde,code,x_model,quality_n=self.enc_model(x_n,name,bx)
                        
                        #if self.best_Model_used[name]["name"]=="samp.-1-0":
                        #    print("w_theta_pred_samples enc",self.best_Model_used[name]["w theta"])
                        #    print("theta_pred_samples enc",theta_tilde,bx)
                        
                        """ 
                        print("name",self.Model_used[name]["name"])
                        print("quality",int(quality_n*2**self.kx_enc),quality_n<=self.quality_n,quality_n,self.quality_n)
                        print("Imin",Imin)
                        print("b model",b_model)
                        print("b bx",self.best_Model_used[name]["b bx"])
                        print("bx",bx)
                        print("Imax",Imax)
                        """
    
                        if quality_n<=self.quality_n   :
                           
                            #print("bx-b_bx",bx-b_bx,2**b_bx-1)
                            #print("self.best_Model_used[name]",self.best_Model_used[name])
                            if bx<=2**self.best_Model_used[name]["b bx"]-1 :
                                #key=name
                                self.best_Model_used[name]["theta tilde"]=theta_tilde
                                self.best_Model_used[name]["code"]=code
                                self.best_Model_used[name]["x model"]=x_model
                                self.best_Model_used[name]["bx"]=bx
                        else :
                            memory_bad_model.append(name)    
                    else :
                        memory_bad_model.append(name)
                    

                    
                if len(memory_bad_model)==len(self.best_Model_used):# aucun modèle n'a plus satisfaire la contrainte de qualité pour ce débit donc on recomence avec un débit plus élevé. 
                    #print("1 memory_bad_model",memory_bad_model,"self.best_Model_used",len(self.best_Model_used))
                    Imin=b_model

            
                    #break
                    
                      
                else:
                    Imax=b_model 
                    #print("2 memory_bad_model",memory_bad_model,"self.best_Model_used",len(self.best_Model_used))
                    for name_ in memory_bad_model:
                        del self.best_Model_used[name_]
                 
                 
        
                b_model=int(np.floor((Imax+Imin)/2))
   
            
            
            
            key,value=self.best_Model_used.popitem()
            self.best_Model_used={key:value}
            #key=next(iter(self.best_Model_used))


            #print("menc 1",self.best_Model_used[key]["m theta"])
            #print("wenc 1",self.best_Model_used[key]["w theta"])
            #print("theta tilde enc 1:",self.best_Model_used[key]["theta tilde"])
            #print("m enc 1:",self.best_Model_used[key]["name"])
            #print("bx enc 1:",self.best_Model_used[key]["bx"])

            
                    
            #for name in self.best_Model_used:
            #    #if self.best_Model_used[name]["family"]=="sin":
            #    print("id : {}".format(name),"name: {},".format(self.best_Model_used[name]["name"]),"RMSE = {:.0f} V,".format(get_rmse(x_n,self.best_Model_used[name]["x model"])*2**self.kx_enc),"b model: {}".format(self.best_Model_used[name]["bx"]+self.best_Model_used[name]["b bx"]))
            
            
            
            if self.best_Model_used[key]["name"]!="none":
                for name in self.Model_used:
                    if self.Model_used[name]["family"]=="pred samples":
                        if self.best_Model_used[key]["family"]!="pred para":
                            self.Model_used[name]["model used"]=key
    
                    elif self.Model_used[name]["family"]=="pred para":
                        if self.best_Model_used[key]["family"]!="pred para":
                            self.Model_used[name]["model used"]=key
                            self.Model_used[name]["m theta"]= self.best_Model_used[key]["theta tilde"]
                        
                            factor= self.Model_used[name]["factor"]
                            self.Model_used[name]["w theta"]= [self.Model_used[key]["w theta"][i]/factor for i in range(len(self.best_Model_used[key]["w theta"]))]
                            self.Model_used[name]["b bx"]= self.best_Model_used[key]["b bx"]
                            #if self.Model_used[name]["name"]=="para.-50":
                            #    print("factor",factor)
                            #    print("self.Model_used[{}][m theta] enc 2".format(self.Model_used[name]["name"]),self.Model_used[name]["m theta"])
                            #    print("self.Model_used[{}][w theta] enc 2".format(self.Model_used[name]["name"]),self.Model_used[name]["w theta"])
                            #    print("self.best_Model_used[id_model][w theta] enc 2",self.Model_used[key]["w theta"])
                    
        
            
            self.id_model_enc=key
            self.id_residual_enc=2

            self.bm_enc=self.bm
            self.bl_enc=0
            
            self.m_enc=self.best_Model_used[key]["name"]
            self.l_enc="none"
            
            self.b_bx_enc=self.best_Model_used[key]["b bx"]
            self.bx_enc=self.best_Model_used[key]["bx"]
            
            self.br_enc=0
            
            
            self.b_kx_enc=self.b_kx
            
            self.b_kr_enc=0
            self.kr_enc=0
            
            
            self.x_model_enc=self.best_Model_used[key]["x model"]*2**(self.kx_enc)
            self.x_residual_enc=np.zeros(self.N)
            self.x_rec_enc=self.x_model_enc
            
            
            
            code_case=[0]   
            #print("code_case",code_case)                       
                        
            code_m=my_bin(key,self.bm_enc)        
            #print("code_m",code_m)
                  
            code_kx=my_bin(self.kx_enc,self.b_kx)
            #print("code_kx",code_kx)
            
            code_bx=my_bin(self.bx_enc,self.b_bx_enc)
            #print("code_bx",code_bx)
            
            code=code_case+code_m+code_kx+code_bx+self.best_Model_used[key]["code"]
            #print("len(code)",len(code),btot)
            #print("code enc",self.best_Model_used[key]["code"])
        
            self.code=code
    
            
            
            return self.code            
            
            
            
            
            
        

                               
            
             
        
        
        
        if not cas_one:

            #Trouver la borne supérieur du budget de bits n_{\text{sup}}^{m} pour coder \boldsymbol{x} pour chaque modèle d'indice m. Pour ce faire, \boldsymbol{x} est passé directement au second étage de compression
            
           
            r_rec,code_r,_=self.enc_residual(x_n,self.quality_n)
            #r_rec,code_r=self.get_r_DWT_BPC_tilde(x_n,self.metric,self.quality_n,self.bmax)
            #key=next(iter(self.best_Model_used))
            #r_rec,code_r=self.get_r_DCT_BPC_tilde(x_n,self.metric,self.quality_n,self.bmax)
            #print("self.nb_coefs_max",self.nb_coefs_max,self.nb_coefs)
            bsup=len(code_r)#np.min([len(code_r),8*self.N])
            bsup_test=bsup
            #quality_sup=get_quality(x_n,r_rec,self.metric)  
            
            
            #print("bsup",bsup,"quality / quality target = {:.2f} / {:.2f}".format(quality_sup,self.quality_n))
            
            
            # On détermine pour chaque valeur de b_x de chaque modèle la meilleur transformation a utiliser pour atteindre avec le moins de bits le critère de qualité 
            
            #key_bx_max=0
            #bx_max=0
            Imax=0
            Imin=0
            
            bad_model=[]
            for id_model in self.best_Model_used:
                
                #if self.best_Model_used[id_model]["bx"]+self.best_Model_used[id_model]["b bx"]>bx_max:
                #    bx_max=self.best_Model_used[id_model]["bx"]+self.best_Model_used[id_model]["b bx"]
                #    key_bx_max=id_model
                
                
                r=x_n-self.best_Model_used[id_model]["x model"]
                
                
                if self.best_Model_used[id_model]["name"]!="none":
                    b_kr=self.b_kr
                    _,kr=normalize(r)
                    if -kr>=2**self.b_kr:
                        kr=-(2**self.b_kr-1)
                    if kr>0:
                        kr=0
                    r_n=r*2**(-kr)
                else :
                    r_n=copy.copy(r)
                    kr=0
                    b_kr=0
                self.best_Model_used[id_model]["b kr"]=b_kr    
                
                
                if self.metric in ["MSE", "RMSE"]:
                    quality_r=self.quality_n*2**(-kr)
                else :
                    quality_r=self.quality_n-self.best_Model_used[id_model]["quality model"]
                #print("quality_r",quality_r*2**self.kx_enc)    
                

               
                r_rec,code_r,id_residual=self.enc_residual(r_n,quality_r)
                #r_rec,code_r=self.get_r_DWT_BPC_tilde(r_n,self.metric,quality_r,self.bmax)
                #id_residual=1
                
              
                #id_residual=next(iter(self.best_Residual_used))

                #if self.best_Residual_used[id_residual]["name"]=="DCT+BPC":
                #    r_rec,code_r=self.get_r_DCT_BPC_tilde(r_n,self.metric,quality_r,self.bmax)

                #elif self.best_Residual_used[id_residual]["name"]=="DWT+BPC":
                #    r_rec,code_r=self.get_r_DWT_BPC_tilde(r_n,self.metric,quality_r,self.bmax)
                #else :
                #    print("la méthode de compression n'hesiste pas")
               
                br=len(code_r)
                #quality_r_get=get_quality(r_n,r_rec,self.metric)  
                quality_get=get_quality(x,self.best_Model_used[id_model]["x model"]*2**self.kx_enc+r_rec*2**(self.kx_enc+kr),self.metric)  
                #print("model",self.best_Model_used[id_model]["name"])
                #print("quality_get",quality_get,"self.quality",self.quality)
                #print(bsup,br)
                
                self.best_Model_used[id_model]["id residual"]=id_residual
                self.best_Model_used[id_model]["name residual"]=self.best_Residual_used[id_residual]['name']
                self.best_Model_used[id_model]["x residual"]=r_rec
                self.best_Model_used[id_model]["br"]=br
                self.best_Model_used[id_model]["code residual"]=code_r
                self.best_Model_used[id_model]["kr"]=kr
                if bsup-br-b_kr>=Imax : # tous les modèles 
                    Imax=np.min([12*8,bsup-br])
                    
                if bsup-br-b_kr<0 or quality_get>self.quality:#or quality_get>self.quality: # ne pas coder ne modèle ne fera pas baisser la qualité de reconstruction
                    bad_model.append(id_model)
                    # dans le meilleur cas on arrive pas à fire baisser le nombre de bits donc on supprimer ce modele
               

                        

                        
                            
                
                #plt.figure(figsize=(8,4), dpi=100)
                #plt.plot(x_n,lw=1,label='x_n')
               # plt.plot(r_n,lw=1,label='r_n')
                #plt.plot(r_rec,lw=1,label='r_rec {}'.format(quality_get))
                #plt.ylabel('Amplitude')
                #plt.legend()
               # plt.grid( which='major', color='#666666', linestyle='-')
                #plt.minorticks_on()
                #plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                #plt.show()  
                
               
               
                #self.best_Model_used[id_model]["id residual"]=id_residual
                #self.best_Model_used[id_model]["name residual"]=self.best_Residual_used[id_residual]["name"]
                #self.best_Model_used[id_model]["br"]=br
                #self.best_Model_used[id_model]["x residual"]=r_rec
                #self.best_Model_used[id_model]["code residual"]=code_r
                
                    
            #print("bad_model",bad_model)
           
            if len(self.best_Model_used)!=len(bad_model):
                for id_model in bad_model:
                   del self.best_Model_used[id_model] 
           
            
      
            Imin=0


          
            #print("self.best_Model_used",self.best_Model_used)
            #print(len(self.best_Model_used))
            #for element in self.best_Model_used:
            #    print(self.best_Model_used[element]["name"],"bx",self.best_Model_used[element]["bx"]+self.best_Model_used[element]["b bx"]) 
            
           
           
            b_model=int(np.floor(Imax/2))
            #print("Imin",Imin,"bmodel", b_model,"Imax",Imax,"dicho_max",np.min([self.stop_dicho,Imax-1]))
           
          
            min_=np.max([Imax-1,1])
            #print("min_,Imin,Imax",min_,Imin,Imax)
            while Imax-Imin>np.min([self.stop_dicho,min_]) :
                #print("stape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print(Imin,b_model,Imax)
                
                #change=False
                memory_bad_model=[]
                nb_model_test=0
                for id_model in  self.best_Model_used:
                    
                    bx=b_model-self.best_Model_used[id_model]["b bx"]-self.best_Model_used[id_model]["b kr"]
                    
                    if (bx>=0):#0 or (bx==0 and self.best_Model_used[id_model]["name"]=="none")):
                        nb_model_test+=1
                        #print(len(x_n))
                        #print(id_model,bx)
                        theta_tilde,code,x_model,quality_n=self.enc_model(x_n,id_model,bx)
                         
                        ##
                        #on encode le résidu
                        ##
                        
                        r=x_n-x_model# définition du résidu
                        
    
                        ########## normalisation de r

                            
                        _,kr=normalize(r)
                        if -kr>=2**self.b_kr:
                            kr=-(2**self.b_kr-1)
                        if kr>0:
                            kr=0
                        r_n=r*2**(-kr)

                            
                    
    
                        
                        #print("quality_n",quality_n)
                        #print("self.best_Model_used[id_model][quality model]",self.best_Model_used[id_model]["quality model"])
                        
                        if metric=="SNR" or metric=="SNR_L1":
                            quality_r_target=(quality-quality_n)#self.best_Model_used[id_model]["quality model"])
                        else :
                            quality_r_target=quality*2**(-self.kx_enc-kr)
                            
                        
                        x_residual,code_residual,id_residual= self.enc_residual(r_n,quality_r_target)
                        #id_residual=self.best_Model_used[id_model]["id residual"]
                        #x_residual,code_residual= self.enc_residual2(r_n,quality_r_target,id_residual)
                        #quality_r=get_quality(r_n,x_residual,'RMSE')*2**(self.kx_enc+kr)
                        #print("quality_n",quality_n)
                        
                        #print("quality<quality_r_target : {:.1f}<{:.1f}".format(quality_r,quality_r_target*2**(self.kx_enc+kr)))
                        #print("br",len(code_residual))
                        #print("id_residual",id_residual)
                            
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(r_n,lw=1,label='r_n')
                        plt.plot(x_residual,lw=1,label='r rec, {} = {:.5f}, target = {} '.format(metric,get_quality(r_n,x_residual,metric),quality_r_target))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show() 
                        """
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(r*2**(self.kx_enc),lw=1,label='r')
                        plt.plot(x_residual*2**(self.kx_enc+kr),lw=1,label='r rec, {} = {:.5f}, target = {:.5f}, code = {} bits  '.format(metric,get_quality(r*2**(self.kx_enc),x_residual*2**(self.kx_enc+kr),metric),quality,len(code_residual)))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        """    
                        
                        
                        
                        #btot_test=self.b_kx+self.bm+b_bx_test+bx_test+self.b_kr+self.bl+len(code_residual)
                

                        #print("b model {}, bx {} + b bx {} + b kr {}".format(b_model,bx,self.best_Model_used[id_model]["b bx"],self.best_Model_used[id_model]["b kr"]),"name :",self.best_Model_used[id_model]["name"],"quality_test / quality = {:.2f} / {:.2f}".format(quality_n*2**self.kx_enc,self.quality_n*2**self.kx_enc))
                        br=len(code_residual)
                        
                        """
                        quality_test=get_quality(x,x_model*2**(self.kx_enc)+x_residual*2**(self.kx_enc+kr),metric)
                        #print("quality test",np.round(quality_test),"quality target",quality)
                        print("----------------------------------------------------")
                        print("name model",self.Model_used[id_model]["name"])
                        print("quality / target ={}/{}".format(int(quality_test),int(quality)))
                        print("Imin<nx<Imax : {}<{}<{}".format(Imin,b_model,Imax))
                        print("b_model+br={}<bsup={}".format(b_model+br,bsup),"valid :",b_model+br<bsup)
                        #print("b bx",self.best_Model_used[id_model]["b bx"])
                        #print("bx",bx)                        
                        """
                        
                        
                        if b_model+br+self.best_Model_used[id_model]["b kr"]<bsup:
                            
                       
                            # key=id_model
                            #if b_model+br<bsup_test:
                            
                            bsup_test=b_model+br+self.best_Model_used[id_model]["b kr"]
                            if bx<=2**self.best_Model_used[id_model]["b bx"]-1: #le modèle n'est sauver que si bx est respécté'
                                
                                
                                self.best_Model_used[id_model]["theta tilde"]=theta_tilde
                                self.best_Model_used[id_model]["code"]=code
                                self.best_Model_used[id_model]["x model"]=x_model
                                self.best_Model_used[id_model]["quality model"]=quality_n
                                self.best_Model_used[id_model]["bx"]=bx
                                 
                                self.best_Model_used[id_model]["id residual"]=id_residual
                                self.best_Model_used[id_model]["name residual"]=self.best_Residual_used[id_residual]['name']
                                self.best_Model_used[id_model]["x residual"]=x_residual
                                self.best_Model_used[id_model]["br"]=br
                                self.best_Model_used[id_model]["code residual"]=code_residual
                                self.best_Model_used[id_model]["kr"]=kr
                            
                        else :
                            memory_bad_model.append(id_model)
                    else :
                        memory_bad_model.append(id_model)
                #print("Imin,Imax",Imin,Imax)
                #print("memory_bad_model",memory_bad_model,"len(self.best_Model_used)",len(self.best_Model_used),nb_model_test)  
                #print("memory_m_not_good",memory_bad_model)
                if len(memory_bad_model)==len(self.best_Model_used):#:# aucun modèle testé n'a plus satisfaire la contrainte de qualité pour ce débit donc on recomence avec un débit plus élevé. 
                    Imin=b_model
                    #break

                else:
                    Imax=b_model 
                    
                    #if len(memory_bad_model)!=len(self.best_Model_used):
                    for name in memory_bad_model:
                        del self.best_Model_used[name]
             
                b_model=int(np.floor((Imax+Imin)/2))
                bsup=bsup_test
                
            """
            for element in self.best_Model_used:
                print(self.best_Model_used[element]["name"],"bx",self.best_Model_used[element]["bx"]+self.best_Model_used[element]["b bx"],"br",self.best_Model_used[element]["br"],"btot",self.best_Model_used[element]["bx"]+self.best_Model_used[element]["b bx"]+self.best_Model_used[element]["br"],self.best_Model_used[element]["id residual"]) 
            
            """
            
            
            
            key,value=self.best_Model_used.popitem()
            self.best_Model_used={key:value}
            #key=next(iter(self.best_Model_used))
            #print("m :",self.best_Model_used[key]["name"])
            #print("bx :",self.best_Model_used[key]["bx"])
            #print("theta tilde :",self.best_Model_used[key]["theta tilde"])
            
            #print(self.best_Model_used[key])
            if self.best_Model_used[key]["name"]!="none":
                for name in self.Model_used:
                    if self.Model_used[name]["family"]=="pred samples":
                        if self.best_Model_used[key]["family"]!="pred para":
                            self.Model_used[name]["model used"]=key
    
                    elif self.Model_used[name]["family"]=="pred para":
                        if self.best_Model_used[key]["family"]!="pred para" :
                            self.Model_used[name]["model used"]=key
                            self.Model_used[name]["m theta"]= self.best_Model_used[key]["theta tilde"]
                            
                            factor= self.Model_used[name]["factor"]
                            self.Model_used[name]["w theta"]= [self.Model_used[key]["w theta"][i]/factor for i in range(len(self.best_Model_used[key]["w theta"]))]
                            self.Model_used[name]["b bx"]= self.best_Model_used[key]["b bx"]
                    


            self.id_model_enc=key
            self.id_residual_enc=self.best_Model_used[key]["id residual"]

            self.bm_enc=self.bm
            self.bl_enc=self.bl
            
            self.m_enc=self.best_Model_used[key]["name"]
            self.l_enc=self.best_Model_used[key]["name residual"]
            
            self.b_bx_enc=self.best_Model_used[key]["b bx"]
            self.bx_enc=self.best_Model_used[key]["bx"]
            
            self.br_enc=self.best_Model_used[key]["br"]
            
            
            self.b_kx_enc=self.b_kx
            
            self.b_kr_enc=self.best_Model_used[key]["b kr"]
            self.kr_enc=self.best_Model_used[key]["kr"]
            
            
            self.x_model_enc=self.best_Model_used[key]["x model"]*2**(self.kx_enc)
            self.x_residual_enc=self.best_Model_used[key]["x residual"]*2**(self.kx_enc+self.kr_enc)
            
            self.x_rec_enc=self.x_model_enc+self.x_residual_enc                
            #print("RMSE tot",int(get_quality(x, self.x_rec_enc, self.metric))  )
                
            """
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(x,lw=1,label='x')
            plt.plot(self.x_rec_enc,lw=1,label='x rec RMSE {}'.format(int(get_quality(x, self.x_rec_enc, self.metric))))
            plt.xlabel('ind')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
            """
            
    
            #print("self.bx_pred_para_max enc",self.bx_pred_para_max)
                                    
            code_case=[1]   
            #print("code_case",code_case)           
                                    
            code_m=my_bin(key,self.bm)        
            #print("code_m",code_m)
                  
            code_kx=my_bin(self.kx_enc,self.b_kx)
            #print("code_kx",code_kx)

            code_bx=my_bin(self.bx_enc,self.b_bx_enc)
            #print("code_bx",code_bx)
            
            code_kr=my_bin(-self.kr_enc,self.b_kr_enc)
            #print("code_kr",code_kr)
            
            
            code_l=my_bin(self.best_Model_used[key]["id residual"],self.bl)
            #print("code_l",code_l)
            
    
            code=code_case+code_m+code_kx+code_bx+self.best_Model_used[key]["code"]+code_kr+code_l+self.best_Model_used[key]["code residual"]
            #print("len(code)",len(code),btot)
            
            
            self.code=code
            #print("code",len(self.code))
            
            return code
                        







class Decode_one_window(Model_Decoder,Residual_Decoder):
    def __init__(self,fn=50,fs=6400, N=128,Model_used={},Residual_used={},verbose=False):
        
        
    
        self.Model_used=Model_used # dictionnaire des modèles utilisés ainsi que leurs caractéristiques respéctives
        self.Residual_used=Residual_used

        
        Model_Decoder.__init__(self,fn,fs,N,False) 
        Residual_Decoder.__init__(self,N)   
        
        
        
        
        ##################### budget de bits servant à décoder le signal
        self.bm=int(np.ceil(np.log2(len(Model_used)))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin ++samples+para+none
        
        self.bl=1  # nombre de bits pour coder la méthode de résidu actuelement: DCT, DWT
           
        self.b_kx=5 # nombre de bits pour coder kx, 0 si modèle pred_samples
        
        self.b_kr=4 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        
        
        
        

        
        #################### grandeurs optimals meilleurs modèle + meilleur méthode de compression de résidu

    def ini_MMC_dec(self,id_model):

        self.best_Model_used={id_model:copy.copy(self.Model_used[id_model])}#copy.deepcopy(self.Model_used)#
        #self.best_Model_used[id_model]=self.Model_used[id_model]
        #self.best_Residual_used=copy.deepcopy(self.Residual_used)
    


    
    def dec_model(self,id_model,code,x_p):
        
        bx=len(code)
        if self.best_Model_used[id_model]["family"]=="pred samples":
            
            id_previous_model=self.best_Model_used[id_model]["model used"]

            
            if self.Model_used[id_previous_model]["name"]!="pred samples":
                self.best_Model_used[id_model]["m theta"]=self.get_m_theta_pred_samples(self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"],0,[0]*self.best_Model_used[id_model]["order"],[np.infty]*self.best_Model_used[id_model]["order"]) 
            else :   
                X_pred_samples2=self.get_X(x_p[0:2*self.N]*2**(-self.kx_dec),self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"]) 
                self.best_Model_used[id_model]["m theta"]=self.get_theta_pred_samples(X_pred_samples2,x_p[2*self.N:3*self.N]*2**(-self.kx_dec),[0]*self.best_Model_used[id_model]["order"],[10]*self.best_Model_used[id_model]["order"]) 
            
            #print("m_theta_pred_samples dec",self.best_Model_used[id_model]["m theta"])
            
            self.best_Model_used[id_model]["X"]=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"])


            self.best_Model_used[id_model]["theta tilde"]= self.get_theta_pred_samples_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_pred_samples(self.best_Model_used[id_model]["X"],*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec            
        
            
        
            
            #print("w_theta_pred_samples dec",self.best_Model_used[id_model]["w theta"])
            #print("theta_pred_samples dec",self.best_Model_used[id_model]["theta tilde"])
            
 
         
        elif self.best_Model_used[id_model]["family"]=="pred para":
            id_previous_model=self.Model_used[id_model]["model used"]
            
            #print(self.Model_used[id_previous_model]["family"])
          
            if self.Model_used[id_previous_model]["family"]=="sin":

                #self.best_Model_used[id_model]["m theta"]=self.Model_used[id_previous_model]["m theta"]
                #self.best_Model_used[id_model]["w theta"]=[self.Model_used[id_previous_model]["w theta"][i]/self.best_Model_used[id_model]["factor"] for i in range(3)]
                
                self.best_Model_used[id_model]["theta tilde"]=self.get_theta_sin_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec  
           
                
            elif self.Model_used[id_previous_model]["family"]=="pred samples":
                
                
  
                
                
                self.best_Model_used[id_model]["X"]=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),self.Model_used[id_previous_model]["order"],self.Model_used[id_previous_model]["eta"])

    
                X=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),self.Model_used[id_previous_model]["order"],self.Model_used[id_previous_model]["eta"])



                #self.best_Model_used[id_model]["m theta"]=self.Model_used[id_previous_model]["theta tilde"]
                #self.best_Model_used[id_model]["w theta"]=[self.Model_used[id_previous_model]["w theta"][i]/self.best_Model_used[id_model]["factor"] for i in range(self.Model_used[id_previous_model]["order"])]

                self.best_Model_used[id_model]["theta tilde"]= self.get_theta_pred_samples_tilde(code,bx,self.best_Model_used[id_model]["m theta"], self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_pred_samples(X,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec                
                
                
            elif self.Model_used[id_previous_model]["family"]=="poly":
                #print("model", self.best_Model_used[id_model])
                #print("model previous", self.Model_used[id_previous_model])
                
                #self.best_Model_used[id_model]["m theta"]=self.Model_used[id_previous_model]["theta tilde"]
            
                #self.best_Model_used[id_model]["w theta"]=[self.Model_used[id_previous_model]["w theta"][i]/self.best_Model_used[id_model]["factor"] for i in range(len(self.Model_used[id_previous_model]["w theta"]))]
                self.best_Model_used[id_model]["theta tilde"]=self.get_theta_poly_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec  
              
                  
        elif self.best_Model_used[id_model]["family"]=="sin":
            
            
            self.best_Model_used[id_model]["theta tilde"]=self.get_theta_sin_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec 
           
      
            
            
            
        elif self.best_Model_used[id_model]["family"]=="poly":
            
     

     
            self.best_Model_used[id_model]["theta tilde"]=self.get_theta_poly_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec   

        
        elif self.best_Model_used[id_model]["family"]=="none": 
            
            self.best_Model_used[id_model]["theta tilde"]=[]
            self.best_Model_used[id_model]["x model"]=np.zeros(self.N)
         
        else :
            print("le modèle {} n'existe pas".format(id_model))
       

        #return theta_tilde,x_rec
        
    
    
    
    def dec_residual(self,id_residual,code_r):


        if self.Residual_used[id_residual]["name"]=="DCT+BPC":
            
            return self.get_r_DCT_BPC_tilde(code_r)
           
        elif self.Residual_used[id_residual]["name"]=="DWT+BPC":
            return self.get_r_DWT_BPC_tilde(code_r)
    
        else :
            return "la méthode est fausse"
                
  
        
        

    def MMC_dec(self,code,x_p):

        
        
        

        #decodage cas
        
        ptr=0
        case=code[ptr]
        if case==0:
            case_one=True
        else :
            case_one=False
        ptr+=1  
      

 
        id_model=int(my_inv_bin(code[ptr:ptr+self.bm]))
        ptr+=self.bm
        #print("id_model", id_model)
        
        
        self.ini_MMC_dec(id_model)
        
        #print(self.best_Model_used)
        

        #print("m_dec", self.best_Model_used[id_model]["name"])
  
        #print("family model", self.best_Model_used[id_model]["family"])    
        
        
        self.kx_dec=int(my_inv_bin(code[ptr:ptr+self.b_kx]))
        ptr+=self.b_kx
        #print("kx dec", self.kx_dec)
         

        #print("b_bx dec",  self.best_Model_used[id_model]["b bx"])

        self.best_Model_used[id_model]["bx"]=int(my_inv_bin(code[ptr:ptr+self.best_Model_used[id_model]["b bx"]]))
        ptr+=self.best_Model_used[id_model]["b bx"]
        #print("bx dec", self.best_Model_used[id_model]["bx"])
        
        
        #print("code dec",code[ptr:ptr+self.best_Model_used[id_model]["bx"]])

        self.dec_model(id_model,code[ptr:ptr+self.best_Model_used[id_model]["bx"]],x_p)

              
        ptr+=self.best_Model_used[id_model]["bx"]
        
        #print("mdec 1",self.best_Model_used[id_model]["m theta"])
        #print("wdec 1",self.best_Model_used[id_model]["w theta"])
        #print("theta tilde dec 1:",self.best_Model_used[id_model]["theta tilde"])
        #print("m dec 1:",self.best_Model_used[id_model]["name"])
        #print("bx dec 1:",self.best_Model_used[id_model]["bx"])
        
        if self.best_Model_used[id_model]["name"]!="none":
            for name in self.Model_used:
                #print(self.Model_used[name]["name"])
                if self.Model_used[name]["family"]=="pred samples":
                    if self.best_Model_used[id_model]["family"]!="pred para":
                        self.Model_used[name]["model used"]=id_model

                elif self.Model_used[name]["family"]=="pred para":
                    if self.best_Model_used[id_model]["family"]!="pred para" :
                        self.Model_used[name]["model used"]=id_model
                    
                        self.Model_used[name]["m theta"]= self.best_Model_used[id_model]["theta tilde"]
                        
                        factor= self.Model_used[name]["factor"]
                        self.Model_used[name]["w theta"]= [self.Model_used[id_model]["w theta"][i]/factor for i in range(len(self.best_Model_used[id_model]["w theta"]))]
                        self.Model_used[name]["b bx"]= self.best_Model_used[id_model]["b bx"]
                        #f self.Model_used[name]["name"]=="para.-50":
                        #    print("factor dec 2", self.Model_used[name]["factor"])
                        #    print("self.Model_used[{}][m theta] dec 2".format(self.Model_used[name]["name"]),self.Model_used[name]["m theta"])
                        #    print("self.Model_used[{}][w theta] dec 2".format(self.Model_used[name]["name"]),self.Model_used[name]["w theta"])
                        #    print("self.best_Model_used[id_model][w theta] dec 2",self.best_Model_used[id_model]["w theta"])
                        #    print("---------------------------------------------------------------------")
            

    
       
        

        if case_one==True:
            

                
            #print(self.best_Model_used[id_model])
            self.x_rec_dec= self.best_Model_used[id_model]["x model"]
        else :
        
    
            if self.best_Model_used[id_model]["name"]!="none":
                self.b_kr_dec=self.b_kr
            else :
                self.b_kr_dec=0
                
            #print("b_kr_dec", self.b_kr_dec)
            
            
            
            self.kr_dec=-int(my_inv_bin(code[ptr:ptr+self.b_kr_dec]))
            ptr+=self.b_kr_dec
            #print("kr", self.kr_dec)
    
            self.best_Model_used[id_model]["id residual"]=int(my_inv_bin(code[ptr:ptr+self.bl]))
            #print("label_residual",label_residual)
    
            self.best_Model_used[id_model]["name residual"]=self.Residual_used[self.best_Model_used[id_model]["id residual"]]["name"]
            ptr+=self.bl
            
            
            self.best_Model_used[id_model]["x residual"]=self.dec_residual(self.best_Model_used[id_model]["id residual"],code[ptr:])*2**(self.kx_dec+self.kr_dec)
            
        
            
            self.x_rec_dec= self.best_Model_used[id_model]["x model"]+self.best_Model_used[id_model]["x residual"]
            
            

            
        