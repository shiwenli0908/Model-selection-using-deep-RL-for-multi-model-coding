# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:41:41 2024

@author: presvotscor
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:17:16 2024

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


class Encode_one_window(Model_Encoder,Residual_Encoder):
    def __init__(self,fn=50,fs=6400, N=128, Model_used={},Residual_used={}, verbose=False):
        

        self.Model_used=Model_used # dictionnaire des modèles utilisés ainsi que leurs caractéristiques respéctives
        self.Residual_used=Residual_used 
        
        
        
        ###########################  Allocation de bits 

        Model_Encoder.__init__(self,fn,fs,N,False) 
        Residual_Encoder.__init__(self,N)   
        

    
        
        ##################### budget de bits servant à décoder le signal

        
        self.bm=int(np.ceil(np.log2(len(Model_used)))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin ++samples+para+none
        
        self.bl=1  # nombre de bits pour coder la méthode de résidu actuelement: DCT, DWT
       
        self.b_kx=5 # nombre de bits pour coder kx, 0 si modèle pred_samples
        
        self.b_kr=4 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        

    def ini_MMC_enc(self,metric,quality,bmax):
        self.metric=metric
        self.quality=quality
        self.bmax=bmax
        
        self.best_Model_used=copy.deepcopy(self.Model_used)
        self.best_Residual_used=copy.deepcopy(self.Residual_used)
        
        
    def verif_order(self,A, B, C):

        # Vérification de la condition A[i] < B[i] < C[i] pour chaque élément
        #"""
        for i in range(len(A)):
            if not (A[i]-0.5*C[i] <= B[i] <= A[i]+0.5*C[i]):
                return False
        #"""
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
                    #print("m_theta sin",self.best_Model_used[name]["m theta"])
                    #print("theta hat sin",self.best_Model_used[name]["theta hat"])
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
                    #print("m_theta poly",self.best_Model_used[name]["m theta"])
                    #print("theta hat poly",self.best_Model_used[name]["theta hat"])
                    del_model.append(name)
                    #del self.best_Model_used[name]
                    
            elif self.best_Model_used[name]["family"]=="pred samples":

                if self.best_Model_used[name]["model used"]!="pred samples":
                    m_theta_pred_samples=self.get_m_theta_pred_samples(self.best_Model_used[name]["order"],self.best_Model_used[name]["eta"],0,[0]*self.best_Model_used[name]["order"],[10]*self.best_Model_used[name]["order"]) 
                else :   
                    X_pred_samples2=self.get_X(x_p_n[0:2*self.N],self.best_Model_used[name]["order"],self.best_Model_used[name]["eta"]) 
                    m_theta_pred_samples=self.get_theta_pred_samples(X_pred_samples2,x_p_n[2*self.N:3*self.N],[0]*self.best_Model_used[name]["order"],[10]*self.best_Model_used[name]["order"]) 
                
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
                    #print("m_theta pred samples",self.best_Model_used[name]["m theta"])
                    #print("theta hat samples",self.best_Model_used[name]["theta hat"])
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
                        #print("m_theta pred para sin",self.best_Model_used[name]["m theta"])
                        #print("theta hat pred para sin",self.best_Model_used[name]["theta hat"])
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
                        #print("m pred para poly",self.best_Model_used[name]["m theta"])
                        #print("theta hat pred para poly",self.best_Model_used[name]["theta hat"])
                        #print("w hat pred para poly",self.best_Model_used[name]["w theta"])
                        
                        #for k in range(len(theta_poly_hat)):
                        #    print(self.best_Model_used[name]["m theta"][k]-0.5*self.best_Model_used[name]["w theta"][k],self.best_Model_used[name]["theta hat"][k],self.best_Model_used[name]["m theta"][k]+0.5*self.best_Model_used[name]["w theta"][k])
                       
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
                        #print("m_theta pred para pred samples",self.best_Model_used[name]["m theta"])
                        #print("theta hat pred para pred samples",self.best_Model_used[name]["theta hat"])
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
        
        #"""
        for name in del_model:
            del self.best_Model_used[name]
        #print("model to delete",del_model)   
      
        if len(bad_model)!=len(self.best_Model_used): #il y a au moins un bon modèle
            for name in bad_model:
                del self.best_Model_used[name]
            #print("good model",self.best_Model_used)
            return True
        #"""
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
            print("id : {}".format(name),"name: {},".format(self.best_Model_used[name]["name"]),"RMSE = {:.0f} V,".format(get_rmse(x_n,self.best_Model_used[name]["x model"])*2**self.kx_enc),"SNR = {:.0f} V,".format(get_snr(x_n,self.best_Model_used[name]["x model"])),"b model max: {}".format(self.best_Model_used[name]["bx"]+self.best_Model_used[name]["b bx"]))
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

            
      
            key=next(iter(self.best_Model_used))
            Imax=self.best_Model_used[key]["bx"]+self.best_Model_used[key]["b bx"]

           
            bsup=Imax
            b_model=0
            while b_model<=bsup:
                
        

                for name in self.best_Model_used : # on parcours tous les modèles en partant du meilleurs en qualité
                    bx=b_model-self.best_Model_used[name]["b bx"]
                    
                    
                    if (bx>=0 and bx<=2**self.best_Model_used[name]["b bx"]-1 ) :#or (bx>=0 and self.best_Model_used[name]["name"]=="none"):

                        theta_tilde,code,x_model,quality_n=self.enc_model(x_n,name,bx)
                        
                        """ 
                        print("name",self.Model_used[name]["name"])
                        print("quality",int(quality_n*2**self.kx_enc),quality_n<=self.quality_n,quality_n,self.quality_n)
                        print("Imin",Imin)
                        print("b model",b_model)
                        print("b bx",self.best_Model_used[name]["b bx"])
                        print("bx",bx)
                        print("Imax",Imax)
                        """
    
                        if quality_n<=self.quality_n and b_model<bsup  :
                            bsup=b_model
                            #print("bx-b_bx",bx-b_bx,2**b_bx-1)
                            #print("self.best_Model_used[name]",self.best_Model_used[name])
  
                            key=name
                            self.best_Model_used[name]["theta tilde"]=theta_tilde
                            self.best_Model_used[name]["code"]=code
                            self.best_Model_used[name]["x model"]=x_model
                            self.best_Model_used[name]["bx"]=bx

                b_model+=1
            

            
            if self.best_Model_used[key]["name"]!="none":
                for name in self.Model_used:
                    if self.Model_used[name]["family"]=="pred samples":
                        if self.best_Model_used[key]["family"]!="pred para":
                            self.Model_used[name]["model used"]=key
    
                    elif self.Model_used[name]["family"]=="pred para":
                        if self.best_Model_used[key]["family"]!="pred para":
                            #if self.Model_used[name]["name"]=="para.-50":
                            #    print("model test",self.Model_used[name]["name"])
                            #    print("previous model",self.Model_used[self.Model_used[name]["model used"]]["name"])
                            #    print("current model",self.best_Model_used[key]["name"])
                            #    print("current m",self.best_Model_used[key]["m theta"])
                            #    print("current w",self.best_Model_used[key]["w theta"])
                                
                                                    
                            self.Model_used[name]["model used"]=key
 
                            self.Model_used[name]["m theta"]=self.best_Model_used[key]["theta tilde"]
                        
                            factor= self.Model_used[name]["factor"]
                            self.Model_used[name]["w theta"]= [self.Model_used[key]["w theta"][i]/factor for i in range(len(self.best_Model_used[key]["w theta"]))]
                            self.Model_used[name]["b bx"]= self.best_Model_used[key]["b bx"]
                            #if self.Model_used[name]["name"]=="para.-50":
                            #    print("factor",factor)
                            #    print("next m",self.Model_used[name]["m theta"])
                            #    print("next w",self.Model_used[name]["w theta"])
                                #print("self.best_Model_used[id_model][w theta] enc 2",self.Model_used[key]["w theta"])
                            
            
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
            #print(self.x_model_enc)
            
            
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
            
            
            
            ### Truver le meilleur modèleet l'allocation de bits
            
            nb_bits_min=np.infty
            best_model=0
            bx=0
            
            
            for name in self.best_Model_used:
                if self.best_Model_used[name]["family"]=="sin":
                   
                    x_model_hat=self.get_model_sin(self.t,*self.best_Model_used[name]["theta hat"])
                    
                    error_model=x_n- x_model_hat
                    
                    #print("sin, MSE target={:.4f} dB".format(10*np.log10(self.quality_n)))
                    nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_sin(self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"],
                                                                                        error_model,self.quality_n,float)
                    #print("sin, MSE find={:.4f} dB".format(10*np.log10(MSE_model_tot_min_find)))
                    
                    #print("sin, nx find ={} b, nr find={:.2f} b".format(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                    #print("sin nx={:.0f}".format(np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                    #print("sin nr={:.0f}".format(self.N*nx_nr_find[-1]))
                    nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                    #print("sin, ntot est={:.0f}".format(nb_bits_est))
                    
                    if nb_bits_est<nb_bits_min:
                        nb_bits_min=nb_bits_est
                        best_model=name
                        bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                        self.best_Model_used[best_model]["b kr"]=self.b_kr
                        
                      
                elif self.best_Model_used[name]["family"]=="poly":
                    order=self.best_Model_used[name]["order"]
                    x_model_hat=self.get_model_poly(self.t,*self.best_Model_used[name]["theta hat"])
                    
                    error_model=x_n- x_model_hat
                    
                    #print("poly order {}, MSE target={} dB".format(order,10*np.log10(self.quality_n)))
                    nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_poly(self.best_Model_used[name]["w theta"],
                                                                                        error_model,self.quality_n,float)
                    #print("poly order {}, MSE find={} dB".format(order,10*np.log10(MSE_model_tot_min_find)))
                    
                    #print("poly order {}, nx find ={} b, nr find={} b".format(order,nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                    
                    #print("poly-{}, nx={:.0f}".format(order,np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                    #print("poly-{}, nr={:.0f}".format(order,self.N*nx_nr_find[-1]))
                    nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                    #print("poly-{}, ntot est={:.0f}".format(order,nb_bits_est))
                    if nb_bits_est<nb_bits_min:
                        nb_bits_min=nb_bits_est
                        best_model=name
                        bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                                    
                        self.best_Model_used[best_model]["b kr"]=self.b_kr
                        
                elif self.best_Model_used[name]["family"]=="pred samples":
                    order=self.best_Model_used[name]["order"]
                    eta=self.best_Model_used[name]["eta"]
                    x_model_hat=self.get_model_pred_samples(self.best_Model_used[name]["X"],*self.best_Model_used[name]["theta hat"])
                    
                    error_model=x_n- x_model_hat
                    
                    #print("pred samples order {}, MSE target={} dB".format(order,10*np.log10(self.quality_n)))
                    nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_pred_samples(self.best_Model_used[name]["w theta"],eta,x_p_n,
                                                                                        error_model,self.quality_n,float)
                    #print("pred samples order {}, MSE find={} dB".format(order,10*np.log10(MSE_model_tot_min_find)))
                    
                    #print("pred samples order {}, nx find ={} b, nr find={} b".format(order,nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                    
                    #print("pred samples-{}, nx={:.0f}".format(order,np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                    #print("pred samples-{}, nr={:.0f}".format(order,self.N*nx_nr_find[-1]))
                    nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                    #print("pred samples-{}, ntot est={:.0f}".format(order,nb_bits_est))
                    if nb_bits_est<nb_bits_min:
                        nb_bits_min=nb_bits_est
                        best_model=name
                        bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                                    
                        self.best_Model_used[best_model]["b kr"]=self.b_kr
                            
                elif self.best_Model_used[name]["family"]=="pred para":
                    name_model_used=self.best_Model_used[name]["model used"]
                    if self.best_Model_used[name_model_used]["family"]=="sin":
                        x_model_hat=self.get_model_sin(self.t,*self.best_Model_used[name]["theta hat"])
                        
                        error_model=x_n- x_model_hat
                        
                        #print("error_model",error_model)
                        #print("pred para sin, MSE target={:.4f} dB".format(10*np.log10(self.quality_n)))
                        nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_sin(self.best_Model_used[name]["m theta"],self.best_Model_used[name]["w theta"],
                                                                                            error_model,self.quality_n,float)
                        #print("pred para sin, MSE find={:.4f} dB".format(10*np.log10(MSE_model_tot_min_find)))
                        
                        #print("pred para sin, nx find ={} b, nr find={:.2f} b".format(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                        #print("pred para (sin), nx={:.0f}".format(np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                        #print("pred para (sin), nr={:.0f}".format(self.N*nx_nr_find[-1]))
                        nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                        #print("pred para (sin), ntot est={:.0f}".format(nb_bits_est))
                        
                        if nb_bits_est<nb_bits_min:
                            nb_bits_min=nb_bits_est
                            best_model=name
                            bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                            self.best_Model_used[best_model]["b kr"]=self.b_kr
                            
                    elif self.best_Model_used[name_model_used]["family"]=="poly":
                        order=self.best_Model_used[name_model_used]["order"]
                        x_model_hat=self.get_model_poly(self.t,*self.best_Model_used[name]["theta hat"])
                        
                        error_model=x_n- x_model_hat
                        
                        #print("poly order {}, MSE target={} dB".format(order,10*np.log10(self.quality_n)))
                        nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_poly(self.best_Model_used[name]["w theta"],
                                                                                            error_model,self.quality_n,float)
                        #print("poly order {}, MSE find={} dB".format(order,10*np.log10(MSE_model_tot_min_find)))
                        
                        #print("poly order {}, nx find ={} b, nr find={} b".format(order,nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                        
                        #print("pred para (poly-{}), nx={:.0f}".format(order,np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                        #print("pred para (poly-{}), nr={:.0f}".format(order,self.N*nx_nr_find[-1]))
                        nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                        #print("pred para (poly-{}), ntot est={:.0f}".format(order,nb_bits_est))
                        if nb_bits_est<nb_bits_min:
                            nb_bits_min=nb_bits_est
                            best_model=name
                            bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                                        
                            self.best_Model_used[best_model]["b kr"]=self.b_kr
                            
                    elif self.best_Model_used[name_model_used]["family"]=="pred samples":
                        order=self.best_Model_used[name_model_used]["order"]
                        eta=self.best_Model_used[name_model_used]["eta"]
                        x_model_hat=self.get_model_pred_samples(self.best_Model_used[name]["X"],*self.best_Model_used[name]["theta hat"])
                        
                        error_model=x_n- x_model_hat
                        
                        #print("pred samples order {}, MSE target={} dB".format(order,10*np.log10(self.quality_n)))
                        nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_pred_samples(self.best_Model_used[name]["w theta"],eta,x_p_n,
                                                                                            error_model,self.quality_n,float)
                        #print("pred samples order {}, MSE find={} dB".format(order,10*np.log10(MSE_model_tot_min_find)))
                        
                        #print("pred samples order {}, nx find ={} b, nr find={} b".format(order,nx_nr_find[0:len(self.best_Model_used[name]["m theta"])],self.N*nx_nr_find[-1]))
                        
                        #print("pred para (pred samples-{}), nx={:.0f}".format(order,np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])))
                        #print("pred para (pred samples-{}), nr={:.0f}".format(order,self.N*nx_nr_find[-1]))
                        nb_bits_est=np.sum(nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])+self.N*nx_nr_find[-1]+self.best_Model_used[name]["b bx"]+self.b_kr
                        #print("pred para (pred samples-{}), ntot est={:.0f}".format(order,nb_bits_est))
                        if nb_bits_est<nb_bits_min:
                            nb_bits_min=nb_bits_est
                            best_model=name
                            bx=sum( nx_nr_find[0:len(self.best_Model_used[name]["m theta"])])
                                        
                            self.best_Model_used[best_model]["b kr"]=self.b_kr        
                else :
                    #x_model_hat=np.zeros(self.N)
                   
                    error_model=x_n
                    
                    #print("None, MSE target={} dB".format(10*np.log10(self.quality_n)))
                    nx_nr_find,MSE_model_tot_min_find=self.get_nx_nr_constraint_MSE_none(error_model,self.quality_n,float)
                                                                                      
                    #print("None, MSE find={} dB".format(10*np.log10(MSE_model_tot_min_find)))
                    
                    #print("None, nr find={} b".format(nx_nr_find*self.N))
                    
                    #print("None, nx=0")
                    #print("None, nr={:.0f}".format(self.N*nx_nr_find))
                    nb_bits_est=self.N*nx_nr_find
                    #print("Non, ntot est={:.0f}".format(nb_bits_est))
                    if nb_bits_est<nb_bits_min:
                        nb_bits_min=nb_bits_est
                        best_model=name
                        bx=0
                        self.best_Model_used[best_model]["b kr"]=0
                                    
            
                
        
            #print("bx",bx,"nb_bits_min",nb_bits_min)
            
            ##
            #on encode le résidu
            ##
            

            
        
          
            
            bx=np.min([np.floor(bx),2**self.best_Model_used[best_model]["b bx"]-1])
            self.best_Model_used[best_model]["bx"]=bx
     
            self.best_Model_used[best_model]["theta tilde"],self.best_Model_used[best_model]["code"],self.best_Model_used[best_model]["x model"],self.best_Model_used[best_model]["quality model"]=self.enc_model(x_n,best_model,bx)
                
 
            #print("theta tilde enc",self.best_Model_used[best_model]["theta tilde"])
            #print("model enc",self.best_Model_used[best_model]["x model"]*2**self.kx_enc)
                    
                    
            r=x_n-self.best_Model_used[best_model]["x model"]# définition du résidu
                    
                 
            ########## normalisation de r
            
           
            #b_kr=self.b_kr
            _,kr=normalize(r)
            if -kr>=2**self.b_kr:
                kr=-(2**self.b_kr-1)
            if kr>0:
                kr=0
            r_n=r*2**(-kr)
            
          
            
            self.best_Model_used[best_model]["kr"]=kr
   
                    
   
            quality_r_target=quality*2**(-self.kx_enc-kr)
                        
                    
            self.best_Model_used[best_model]["x residual"],self.best_Model_used[best_model]["code residual"],self.best_Model_used[best_model]["id residual"]= self.enc_residual(r_n,quality_r_target)
            self.best_Model_used[best_model]["name residual"]=self.best_Residual_used[self.best_Model_used[best_model]["id residual"]]['name']        

            self.best_Model_used[best_model]["br"]=len(self.best_Model_used[best_model]["code residual"])
            
     
            
            if self.best_Model_used[best_model]["name"]!="none":
                for name in self.Model_used:
                    if self.Model_used[name]["family"]=="pred samples":
                        if self.best_Model_used[best_model]["family"]!="pred para":
                            self.Model_used[name]["model used"]=best_model
    
                    elif self.Model_used[name]["family"]=="pred para":
                        if self.best_Model_used[best_model]["family"]!="pred para" :
                            self.Model_used[name]["model used"]=best_model
                            self.Model_used[name]["m theta"]= self.best_Model_used[best_model]["theta tilde"]
                            
                            factor= self.Model_used[name]["factor"]
                            self.Model_used[name]["w theta"]= [self.Model_used[best_model]["w theta"][i]/factor for i in range(len(self.best_Model_used[best_model]["w theta"]))]
                            self.Model_used[name]["b bx"]= self.best_Model_used[best_model]["b bx"]
                    


            self.id_model_enc=best_model
            self.id_residual_enc=self.best_Model_used[best_model]["id residual"]
            
            self.bm_enc=self.bm
            self.bl_enc=self.bl
            
            self.m_enc=self.best_Model_used[best_model]["name"]
            self.l_enc=self.best_Model_used[best_model]["name residual"]
            
            self.b_bx_enc=self.best_Model_used[best_model]["b bx"]
            self.bx_enc=self.best_Model_used[best_model]["bx"]
            
            self.br_enc=self.best_Model_used[best_model]["br"]
            
            
            self.b_kx_enc=self.b_kx
            
            self.b_kr_enc=self.best_Model_used[best_model]["b kr"]
            self.kr_enc=self.best_Model_used[best_model]["kr"]
            
            
            self.x_model_enc=self.best_Model_used[best_model]["x model"]*2**(self.kx_enc)
            
            self.x_residual_enc=self.best_Model_used[best_model]["x residual"]*2**(self.kx_enc+self.kr_enc)
            
            self.x_rec_enc=self.x_model_enc+self.x_residual_enc                
            #print("RMSE tot",int(get_quality(x, self.x_rec_enc, self.metric)))
                
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
                                    
            code_m=my_bin(best_model,self.bm)        
            #print("code_m",code_m)
                  
            code_kx=my_bin(self.kx_enc,self.b_kx)
            #print("code_kx",code_kx)

            code_bx=my_bin(self.bx_enc,self.b_bx_enc)
            #print("code_bx",code_bx)
            
            code_kr=my_bin(-self.kr_enc,self.b_kr_enc)
            #print("code_kr",code_kr)
            
            
            code_l=my_bin(self.best_Model_used[best_model]["id residual"],self.bl)
            #print("code_l",code_l)
            
    
            code=code_case+code_m+code_kx+code_bx+self.best_Model_used[best_model]["code"]+code_kr+code_l+self.best_Model_used[best_model]["code residual"]
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

    def ini_MMC_dec(self,id_model,bmax):

        self.best_Model_used={id_model:copy.copy(self.Model_used[id_model])}#copy.deepcopy(self.Model_used)#
        #self.best_Model_used[id_model]=self.Model_used[id_model]
        #self.best_Residual_used=copy.deepcopy(self.Residual_used)
        self.bmax=bmax


    
    def dec_model(self,id_model,code,x_p):
        
        bx=len(code)
        if self.best_Model_used[id_model]["family"]=="pred samples":
            
            id_previous_model=self.best_Model_used[id_model]["model used"]

            
            if self.Model_used[id_previous_model]["name"]!="pred samples":
                self.best_Model_used[id_model]["m theta"]=self.get_m_theta_pred_samples(self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"],0,[0]*self.best_Model_used[id_model]["order"],[10]*self.best_Model_used[id_model]["order"]) 
            else :   
                X_pred_samples2=self.get_X(x_p[0:2*self.N]*2**(-self.kx_dec),self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"]) 
                self.best_Model_used[id_model]["m theta"]=self.get_theta_pred_samples(X_pred_samples2,x_p[2*self.N:3*self.N]*2**(-self.kx_dec),self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"]) 
            
            self.best_Model_used[id_model]["X"]=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"])


            self.best_Model_used[id_model]["theta tilde"]= self.get_theta_pred_samples_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_pred_samples(self.best_Model_used[id_model]["X"],*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec            
        
            
        
            
            
            
 
         
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
           
            #print("theta tilde dec",self.best_Model_used[id_model]["theta tilde"])
            #print("model dec",self.best_Model_used[id_model]["x model"])
            
            
            
        elif self.best_Model_used[id_model]["family"]=="poly":
            
     

     
            self.best_Model_used[id_model]["theta tilde"]=self.get_theta_poly_tilde(code,bx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec   
            
            #print("theta tilde dec",self.best_Model_used[id_model]["theta tilde"])
        
        elif self.best_Model_used[id_model]["family"]=="none": 
            
            self.best_Model_used[id_model]["theta tilde"]=[]
            self.best_Model_used[id_model]["x model"]=np.zeros(self.N)
         
        else :
            print("le modèle {} n'existe pas".format(id_model))
       

        #return theta_tilde,x_rec
        
    
    
    
    def dec_residual(self,id_residual,code_r):


        if self.Residual_used[id_residual]["name"]=="DCT+BPC":
            
            return self.get_r_DCT_BPC_tilde(code_r,self.bmax)
           
        elif self.Residual_used[id_residual]["name"]=="DWT+BPC":
            return self.get_r_DWT_BPC_tilde(code_r,self.bmax)
    
        else :
            return "la méthode est fausse"
                
  
        
        

    def MMC_dec(self,code,x_p,bmax):

        
        
        

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
        
        
        self.ini_MMC_dec(id_model,bmax)
        
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
                    
                        self.Model_used[name]["m theta"]=self.best_Model_used[id_model]["theta tilde"]
                        
                        factor= self.Model_used[name]["factor"]
                        self.Model_used[name]["w theta"]= [self.Model_used[id_model]["w theta"][i]/factor for i in range(len(self.best_Model_used[id_model]["w theta"]))]
                        self.Model_used[name]["b bx"]= self.best_Model_used[id_model]["b bx"]
                        #if self.Model_used[name]["name"]=="para.-50":
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
            #print("label_residual",self.best_Model_used[id_model]["id residual"])
    
            self.best_Model_used[id_model]["name residual"]=self.Residual_used[self.best_Model_used[id_model]["id residual"]]["name"]
            ptr+=self.bl
            
            
            self.best_Model_used[id_model]["x residual"]=self.dec_residual(self.best_Model_used[id_model]["id residual"],code[ptr:])*2**(self.kx_dec+self.kr_dec)
            
        
            
            self.x_rec_dec= self.best_Model_used[id_model]["x model"]+self.best_Model_used[id_model]["x residual"]
            
            

            