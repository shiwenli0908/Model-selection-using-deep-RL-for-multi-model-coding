# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:17:16 2024

@author: presvotscor
"""


import numpy as np
import copy

from Normalize import normalize

from codage_model import Model_Encoder,Model_Decoder
from codage_residu import Residual_Encoder,Residual_Decoder

from Measures import get_quality,my_bin,my_inv_bin
import matplotlib.pyplot as plt

class Encode_one_window(Model_Encoder,Residual_Encoder):
    def __init__(self,fn=50,fs=6400, N=128, Model_used={},Residual_used={}):
        
        self.Model_used=Model_used # Set of model used
        self.Residual_used=Residual_used # set of residual used
        
        Model_Encoder.__init__(self,fn,fs,N,False) 
        Residual_Encoder.__init__(self,N)   
        
        ##################### initilisation of header
        
        self.nm=int(np.ceil(np.log2(len(Model_used)+0.1))) # number of bits to encode the number of models
        self.nl=int(np.ceil(np.log2(len(Residual_used)+0.1)))  # number of bits to encode the residual
        self.n_kx=5 # number of bits to encode kx
        self.n_kr=4  # number of bits to encode kr, 0 if model used is none
        
        self.nb_max_bit_theta=16
        
    def ini_MMC_enc(self):
        
        self.best_Model_used=copy.deepcopy(self.Model_used)
        self.best_Residual_used=copy.deepcopy(self.Residual_used)
    
    def get_header(self,x,x_p):
        _,kx=normalize(x)
        if kx>=2**self.n_kx:
            kx=2**self.n_kx-1
        if kx<0:
            kx==0
       
            
        _,kx_p=normalize(x_p[self.N:])
        if kx_p>=2**self.n_kx:
            kx_p=2**self.n_kx-1
        if kx_p<0:
            kx_p==0    
            
        for id_model in self.Model_used:
            if self.best_Model_used[id_model]["family"]=="sin":
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"]=self.n_kx
                self.best_Model_used[id_model]["n kr"]=self.n_kr
                self.best_Model_used[id_model]["kx"]=kx
                self.best_Model_used[id_model]["xn"]=x*2**(-kx)
                
            elif self.best_Model_used[id_model]["family"]=="poly":
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.best_Model_used[id_model]["order"]+1)*self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"]=self.n_kx
                self.best_Model_used[id_model]["n kr"]=self.n_kr
                self.best_Model_used[id_model]["kx"]=kx
                self.best_Model_used[id_model]["xn"]=x*2**(-kx)
                
            elif self.best_Model_used[id_model]["family"]=="pred samples":
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.best_Model_used[id_model]["order"])*self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"]=self.n_kx
                self.best_Model_used[id_model]["n kr"]=self.n_kr
                self.best_Model_used[id_model]["kx"]=kx
                self.best_Model_used[id_model]["xn"]=x*2**(-kx)
                self.best_Model_used[id_model]["xn previous"]=x_p*2**(-kx)
                
            elif self.best_Model_used[id_model]["family"]=="pred para":
                id_previous_model=self.best_Model_used[id_model]["model used"]
                self.best_Model_used[id_model]["n kx"]=self.n_kx
                self.best_Model_used[id_model]["n kr"]=self.n_kr
                self.best_Model_used[id_model]["kx"]=kx
                self.best_Model_used[id_model]["xn"]=x*2**(-kx)
                self.best_Model_used[id_model]["xn previous"]=x_p*2**(-kx)
                if self.Model_used[id_previous_model]["family"]=="sin":
                    self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
                    
                elif self.Model_used[id_previous_model]["family"]=="poly":
                    self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.best_Model_used[id_previous_model]["order"]+1)*self.nb_max_bit_theta)))                     
                
                elif self.Model_used[id_previous_model]["family"]=="pred samples":
                    self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.best_Model_used[id_previous_model]["order"])*self.nb_max_bit_theta)))
                   
            elif self.best_Model_used[id_model]["family"]=="none":   
                self.best_Model_used[id_model]["n nx"]=0
                self.best_Model_used[id_model]["n kx"]=self.n_kx
                self.best_Model_used[id_model]["n kr"]=0   
                self.best_Model_used[id_model]["kx"]=kx
                self.best_Model_used[id_model]["xn"]=x*2**(-kx)
        return kx,kx_p                       
               
        
    def get_theta(self):
        for id_model in self.Model_used:
            
            if self.best_Model_used[id_model]["family"]=="sin":
                theta_sin_hat=self.get_theta_sin(self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["theta hat"]=theta_sin_hat

            elif self.best_Model_used[id_model]["family"]=="poly":
                theta_poly_hat=self.get_theta_poly(self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_model]["order"])
                self.best_Model_used[id_model]["theta hat"]=theta_poly_hat

            elif self.best_Model_used[id_model]["family"]=="pred samples":
                m_theta_pred_samples=self.get_m_theta_pred_samples(self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"],0,[0]*self.best_Model_used[id_model]["order"],[10]*self.best_Model_used[id_model]["order"]) 
                self.best_Model_used[id_model]["m theta"]=m_theta_pred_samples
                
                X_pred_samples=self.get_X(self.best_Model_used[id_model]["xn previous"],self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"])
                theta_pred_samples_hat=self.get_theta_pred_samples(X_pred_samples,self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                
                self.best_Model_used[id_model]["X"]=X_pred_samples
                self.best_Model_used[id_model]["theta hat"]=theta_pred_samples_hat

            elif self.best_Model_used[id_model]["family"]=="pred para":
                id_previous_model=self.best_Model_used[id_model]["model used"]
                if self.Model_used[id_previous_model]["family"]=="sin":
                    theta_sin_hat=self.get_theta_sin(self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                    self.best_Model_used[id_model]["theta hat"]=theta_sin_hat

                elif self.Model_used[id_previous_model]["family"]=="poly":
                    theta_poly_hat=self.get_theta_poly(self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_previous_model]["order"])
                    self.best_Model_used[id_model]["theta hat"]=theta_poly_hat 

                elif self.Model_used[id_previous_model]["family"]=="pred samples":
                    X_pred_samples=self.get_X(self.best_Model_used[id_model]["xn previous"],self.best_Model_used[id_previous_model]["order"],self.best_Model_used[id_previous_model]["eta"])
                    theta_pred_samples_hat=self.get_theta_pred_samples(X_pred_samples,self.best_Model_used[id_model]["xn"],self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                    self.best_Model_used[id_model]["X"]=self.best_Model_used[id_previous_model]["X"]
                    self.best_Model_used[id_model]["theta hat"]=theta_pred_samples_hat

            elif self.best_Model_used[id_model]["family"]=="none":
                self.best_Model_used[id_model]["theta hat"]=[]

            else:     
                print("error: the model {} does not exist".format(id_model))
        

    def enc_model(self,id_model,nx):
        if self.best_Model_used[id_model]["family"]=="pred samples":
            theta_tilde,code_theta_tilde=self.get_theta_pred_samples_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            x_rec=self.get_model_pred_samples(self.best_Model_used[id_model]["X"],*theta_tilde) 

        elif self.best_Model_used[id_model]["family"]=="pred para":
            id_previous_model=self.best_Model_used[id_model]["model used"]
            if self.Model_used[id_previous_model]["family"]=="sin":
                theta_tilde,code_theta_tilde=self.get_theta_sin_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                x_rec=self.get_model_sin(self.t,*theta_tilde) 
                
            elif self.Model_used[ id_previous_model]["family"]=="pred samples":
                theta_tilde,code_theta_tilde=self.get_theta_pred_samples_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                X=self.best_Model_used[id_model]["X"]
                x_rec=self.get_model_pred_samples(X,*theta_tilde) 
               
            elif self.Model_used[id_previous_model]["family"]=="poly":
                theta_tilde,code_theta_tilde=self.get_theta_poly_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                x_rec=self.get_model_poly(self.t,*theta_tilde) 
                  
        elif self.best_Model_used[id_model]["family"]=="sin":
            theta_tilde,code_theta_tilde=self.get_theta_sin_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            x_rec=self.get_model_sin(self.t,*theta_tilde) 
             
        elif self.best_Model_used[id_model]["family"]=="poly":
            theta_tilde,code_theta_tilde=self.get_theta_poly_tilde(self.best_Model_used[id_model]["theta hat"],nx,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            x_rec=self.get_model_poly(self.t,*theta_tilde) 
        
        elif self.best_Model_used[id_model]["family"]=="none": 
            theta_tilde=[]
            code_theta_tilde=[]
            x_rec=np.zeros(self.N)
        else :
            print("error: the model {} does not exist".format(id_model))
       
        return theta_tilde,code_theta_tilde,x_rec
        
    
    def enc_residual(self,id_residual,r,n_r):

        if self.best_Residual_used[id_residual]["name"]=="DCT+BPC":
            r_rec,code_r=self.get_r_DCT_BPC_tilde(r,"SNR",-np.infty,n_r)
 
        elif self.best_Residual_used[id_residual]["name"]=="DWT+BPC":
            r_rec,code_r=self.get_r_DWT_BPC_tilde(r,"SNR",-np.infty,n_r)

        elif self.best_Residual_used[id_residual]["name"]=="none":
            r_rec=np.zeros(self.N)
            code_r=[]
        else :
            print("error: the method {} does not exist".format(id_residual))
            
        return r_rec,code_r
    

    def MMC_enc(self,x,x_previous,id_model,id_residual,n_x,n_max):
     
        self.ini_MMC_enc()

        kx,kx_previous=self.get_header(x,x_previous)
           
        self.get_theta()
        

        ##
        # first stage
        ##
                
        ##################### get theta for each model
        if self.best_Model_used[id_model]["name"]=="none":
            n_x=0
        else:
            n_x=np.min([n_x,self.nb_max_bit_theta*len(self.best_Model_used[id_model]["w theta"])-1])
       
        theta_tilde,code_model,x_model=self.enc_model(id_model,n_x)
        SNR_model=-get_quality(self.best_Model_used[id_model]["xn"],x_model,"SNR")
 
        
        ##
        # second stage
        ##             
       
        r=self.best_Model_used[id_model]["xn"]-x_model # residual
        
        ##################### normalization of r
    
        _,kr=normalize(r)
        if -kr>=2**self.best_Model_used[id_model]["n kr"]:
            kr=-(2**self.best_Model_used[id_model]["n kr"]-1)
        if kr>0:
            kr=0
        r_n=r*2**(-kr)
         
    
        ##################### best residual
        n_r=n_max-n_x-self.nm-self.nl-self.n_kx-self.n_kr-self.best_Model_used[id_model]["n nx"]
        
        #print(self.n_kx,self.n_kr,self.best_Model_used[id_model]["n nx"])
        #print("n_r_enc",n_r)
        x_residual,code_residual= self.enc_residual(id_residual,r_n,n_r)
        
        #print("code_residual",code_residual)
        SNR_residual=-get_quality(r_n,x_residual,"SNR")
        
        n_r_reel=len(code_residual)
        
        self.best_Model_used[id_model]["theta tilde"]=theta_tilde
        self.best_Model_used[id_model]["code model"]=code_model
        self.best_Model_used[id_model]["x model"]=x_model
        self.best_Model_used[id_model]["quality model"]=SNR_model
        self.best_Model_used[id_model]["nx"]=n_x
         
        self.best_Model_used[id_model]["id residual"]=id_residual
        self.best_Model_used[id_model]["name residual"]=self.best_Residual_used[id_residual]['name']
        self.best_Model_used[id_model]["x residual"]=x_residual
        self.best_Model_used[id_model]["quality residual"]=SNR_residual
        self.best_Model_used[id_model]["nr"]=n_r_reel
        self.best_Model_used[id_model]["code residual"]=code_residual
 
        self.best_Model_used[id_model]["kr"]=kr
        
        self.best_model_used=id_model
        

        if self.best_Model_used[self.best_model_used]["name"]!="none":
            for id_model in self.Model_used:
                if self.Model_used[id_model]["family"]=="pred samples":
                    if self.best_Model_used[self.best_model_used]["family"]!="pred para":
                        self.Model_used[id_model]["model used"]=self.best_model_used

                elif self.Model_used[id_model]["family"]=="pred para":
                    if self.best_Model_used[self.best_model_used]["family"]!="pred para" :
                        self.Model_used[id_model]["model used"]=self.best_model_used
                        self.Model_used[id_model]["m theta"]= self.best_Model_used[self.best_model_used]["theta tilde"]
                        
                        factor= self.Model_used[id_model]["factor"]
                        self.Model_used[id_model]["w theta"]= [self.Model_used[self.best_model_used]["w theta"][i]/factor for i in range(len(self.best_Model_used[self.best_model_used]["w theta"]))]
                        self.Model_used[id_model]["n nx"]= self.best_Model_used[self.best_model_used]["n nx"]
                

        
        # encode the signal
        self.id_model_enc=self.best_model_used
        self.id_residual_enc=self.best_Model_used[self.best_model_used]["id residual"]
        
        self.nm_enc=self.nm
        self.nl_enc=self.nl
        
        self.m_enc=self.best_Model_used[self.best_model_used]["name"]
        self.l_enc=self.best_Model_used[self.best_model_used]["name residual"]
        
        self.n_nx_enc=self.best_Model_used[self.best_model_used]["n nx"]
        self.nx_enc=self.best_Model_used[self.best_model_used]["nx"]
        self.nr_enc=self.best_Model_used[self.best_model_used]["nr"]

        
        self.n_kx_enc=self.best_Model_used[self.best_model_used]["n kx"]
        self.n_kr_enc=self.best_Model_used[self.best_model_used]["n kr"]
        
        self.kx_enc=self.best_Model_used[self.best_model_used]["kx"]
        self.kr_enc=self.best_Model_used[self.best_model_used]["kr"]
        
        self.x_model_enc=self.best_Model_used[self.best_model_used]["x model"]*2**(self.kx_enc)
        #print("x_model_enc",self.x_model_enc)
        self.x_residual_enc=self.best_Model_used[self.best_model_used]["x residual"]*2**(self.kx_enc+self.kr_enc)
        #print("x_residual_enc",self.x_residual_enc)
        self.x_rec_enc=self.x_model_enc+self.x_residual_enc                


        code_m=my_bin(self.best_model_used,self.nm)        
        #print("code_m",code_m)
              
        code_kx=my_bin(self.kx_enc,self.n_kx_enc)
        #print("code_kx",code_kx,self.kx_enc)

        code_nx=my_bin(self.nx_enc,self.n_nx_enc)
        #print("code_nx",code_nx,self.n_nx_enc)
        
        code_kr=my_bin(-self.kr_enc,self.n_kr_enc)
        #print("code_kr",code_kr,-self.kr_enc)
        
        code_l=my_bin(self.best_Model_used[self.best_model_used]["id residual"],self.nl)
        #print("code_l",code_l)
        
        code=code_m+code_kx+code_nx+self.best_Model_used[self.best_model_used]["code model"]+code_kr+code_l+self.best_Model_used[self.best_model_used]["code residual"]
        #print("len(code)",len(code),btot)
        
        #print("code_model",self.best_Model_used[self.best_model_used]["code model"])
        self.code=code

        return code
                        







class Decode_one_window(Model_Decoder,Residual_Decoder):
    def __init__(self,fn=50,fs=6400, N=128,Model_used={},Residual_used={}):
        
        self.Model_used=Model_used # Set of model used
        self.Residual_used=Residual_used # set of residual used
        
        Model_Decoder.__init__(self,fn,fs,N,False) 
        Residual_Decoder.__init__(self,N)   
        
        ##################### initilisation of header
        
        self.nm=int(np.ceil(np.log2(len(Model_used)+0.1))) # number of bits to encode the number of models
        self.nl=int(np.ceil(np.log2(len(Residual_used)+0.1)))  # number of bits to encode the residual
        self.n_kx=5 # number of bits to encode kx
        self.n_kr=4  # number of bits to encode kr, 0 if model used is none
        
        self.nb_max_bit_theta=16
        
    def ini_MMC_dec(self,id_model):

        self.best_Model_used={id_model:copy.copy(self.Model_used[id_model])}

        
    def dec_header(self,id_model):
        

        if self.best_Model_used[id_model]["family"]=="pred samples":
            self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(self.best_Model_used[id_model]["order"]*self.nb_max_bit_theta)))
            self.best_Model_used[id_model]["n kx"]=self.n_kx
            self.best_Model_used[id_model]["n kr"]=self.n_kr

        
        elif self.best_Model_used[id_model]["family"]=="pred para":
            id_previous_model=self.Model_used[id_model]["model used"]
            self.best_Model_used[id_model]["n kx"]=self.n_kx
            self.best_Model_used[id_model]["n kr"]=self.n_kr

            if self.Model_used[id_previous_model]["family"]=="sin":
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
            
            elif self.Model_used[id_previous_model]["family"]=="pred samples":
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(self.Model_used[id_previous_model]["order"]*self.nb_max_bit_theta)))
            
            elif self.Model_used[id_previous_model]["family"]=="poly":   
                self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.Model_used[id_previous_model]["order"]+1)*self.nb_max_bit_theta)))  
        
        elif self.best_Model_used[id_model]["family"]=="sin":
            self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
            self.best_Model_used[id_model]["n kx"]=self.n_kx
            self.best_Model_used[id_model]["n kr"]=self.n_kr
            
            
        elif self.best_Model_used[id_model]["family"]=="poly":
            self.best_Model_used[id_model]["n nx"]=int(np.ceil(np.log2((self.best_Model_used[id_model]["order"]+1)*self.nb_max_bit_theta)))
            self.best_Model_used[id_model]["n kr"]=self.n_kx
            self.best_Model_used[id_model]["n kr"]=self.n_kr
            
        elif self.best_Model_used[id_model]["family"]=="none":
            self.best_Model_used[id_model]["n nx"]=0
            self.best_Model_used[id_model]["n kx"]=self.n_kr
            self.best_Model_used[id_model]["n kr"]=0
            
    def dec_model(self,id_model,code_m,x_previous_n):
        
        n_x=len(code_m)
        if self.best_Model_used[id_model]["family"]=="pred samples":
            self.best_Model_used[id_model]["m theta"]=self.get_m_theta_pred_samples(self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"],0,[0]*self.best_Model_used[id_model]["order"],[10]*self.best_Model_used[id_model]["order"]) 
            self.best_Model_used[id_model]["X"]=self.get_X(x_previous_n[0:2*self.N],self.best_Model_used[id_model]["order"],self.best_Model_used[id_model]["eta"])
            self.best_Model_used[id_model]["theta tilde"]= self.get_theta_pred_samples_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_pred_samples(self.best_Model_used[id_model]["X"],*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec            

            
        elif self.best_Model_used[id_model]["family"]=="pred para":
            id_previous_model=self.Model_used[id_model]["model used"]
            if self.Model_used[id_previous_model]["family"]=="sin":
                self.best_Model_used[id_model]["theta tilde"]=self.get_theta_sin_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec  
                 
            elif self.Model_used[id_previous_model]["family"]=="pred samples":
                self.best_Model_used[id_model]["X"]=self.get_X(x_previous_n[0:2*self.N],self.Model_used[id_previous_model]["order"],self.Model_used[id_previous_model]["eta"])
                X=self.get_X(x_previous_n[0:2*self.N],self.Model_used[id_previous_model]["order"],self.Model_used[id_previous_model]["eta"])
                self.best_Model_used[id_model]["theta tilde"]= self.get_theta_pred_samples_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"], self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_pred_samples(X,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec                
                
            elif self.Model_used[id_previous_model]["family"]=="poly":
                self.best_Model_used[id_model]["theta tilde"]=self.get_theta_poly_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
                self.best_Model_used[id_model]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec  
                
        elif self.best_Model_used[id_model]["family"]=="sin":
            self.best_Model_used[id_model]["theta tilde"]=self.get_theta_sin_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec 
            
        elif self.best_Model_used[id_model]["family"]=="poly":
            self.best_Model_used[id_model]["theta tilde"]=self.get_theta_poly_tilde(code_m,n_x,self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"])
            self.best_Model_used[id_model]["x model"]=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta tilde"])*2**self.kx_dec   
            
        elif self.best_Model_used[id_model]["family"]=="none": 
            self.best_Model_used[id_model]["theta tilde"]=[]
            self.best_Model_used[id_model]["x model"]=np.zeros(self.N)

         
    def dec_residual(self,id_residual,code_r,n_r):
        if self.Residual_used[id_residual]["name"]=="DCT+BPC":
            return self.get_r_DCT_BPC_tilde(code_r,n_r)*2**(self.kx_dec+self.kr_dec)
           
        elif self.Residual_used[id_residual]["name"]=="DWT+BPC":
            return self.get_r_DWT_BPC_tilde(code_r,n_r)*2**(self.kx_dec+self.kr_dec)
        
        elif self.Residual_used[id_residual]["name"]=="none":
            return np.zeros(self.N)
        else :
            print("error dec: residual method")
                
  
        
    def MMC_dec(self,code,x_previous,n_max):
        ptr=0

        id_model=int(my_inv_bin(code[ptr:ptr+self.nm]))
        ptr+=self.nm
        #print("id_model", id_model)
        
        self.ini_MMC_dec(id_model)
        
        """
        if self.best_Model_used[id_model]["family"]=="pred para" or self.best_Model_used[id_model]["family"]=="pred samples":
            _,kx_p=normalize(x_previous[self.N:2*self.N])
            if kx_p>=2**self.n_kx:
                kx_p=2**self.n_kx-1
            if kx_p<0:
                kx_p==0  
            self.kx_dec=kx_p
        else :
        """
        self.kx_dec=int(my_inv_bin(code[ptr:ptr+self.n_kx]))
        
        #print("kx dec",code[ptr:ptr+self.n_kx], self.kx_dec)
        ptr+=self.n_kx
        
        self.dec_header(id_model)
        #print("n nx dec",self.best_Model_used[id_model]["n nx"])
        
        
        
        
        self.best_Model_used[id_model]["nx"]=int(my_inv_bin(code[ptr:ptr+self.best_Model_used[id_model]["n nx"]]))
        
        #print("nx dec", self.best_Model_used[id_model]["nx"],code[ptr:ptr+self.best_Model_used[id_model]["n nx"]])
        ptr+=self.best_Model_used[id_model]["n nx"]
        
        #print("theta dec",code[ptr:ptr+self.best_Model_used[id_model]["nx"]])
        self.dec_model(id_model,code[ptr:ptr+self.best_Model_used[id_model]["nx"]],x_previous*2**(-self.kx_dec))
        ptr+=self.best_Model_used[id_model]["nx"]
        
        self.kr_dec=-int(my_inv_bin(code[ptr:ptr+self.best_Model_used[id_model]["n kr"]]))
        
        #print("code kr dec",code[ptr:ptr+self.best_Model_used[id_model]["n kr"]], self.kr_dec)
        ptr+=self.best_Model_used[id_model]["n kr"]
        
        self.best_Model_used[id_model]["id residual"]=int(my_inv_bin(code[ptr:ptr+self.nl]))
        ptr+=self.nl

        self.best_Model_used[id_model]["name residual"]=self.Residual_used[self.best_Model_used[id_model]["id residual"]]["name"]
        #print("residual used",self.best_Model_used[id_model]["name residual"])
        
        #print("code_residual_dec",code[ptr:])
        n_r=n_max-self.best_Model_used[id_model]["n nx"]-self.best_Model_used[id_model]["nx"]-self.n_kx-self.n_kr-self.nm-self.nl
        #print("n_r dec",n_r,self.kx_dec,self.kr_dec,self.best_Model_used[id_model]["n nx"])
        
        self.best_Model_used[id_model]["x residual"]=self.dec_residual(self.best_Model_used[id_model]["id residual"],code[ptr:],n_r)
        
        #print(self.best_Model_used[id_model]["x model"])
        self.x_rec_dec= self.best_Model_used[id_model]["x model"]+self.best_Model_used[id_model]["x residual"]
        
        #print("x_model_dec",self.best_Model_used[id_model]["x model"])
        #print("x_residual_dec",self.best_Model_used[id_model]["x residual"])
        
        if self.best_Model_used[id_model]["name"]!="none":
            for name in self.Model_used:
                if self.Model_used[name]["family"]=="pred samples":
                    if self.best_Model_used[id_model]["family"]!="pred para":
                        self.Model_used[name]["model used"]=id_model

                elif self.Model_used[name]["family"]=="pred para":
                    if self.best_Model_used[id_model]["family"]!="pred para" :
                        self.Model_used[name]["model used"]=id_model
                    
                        self.Model_used[name]["m theta"]=self.best_Model_used[id_model]["theta tilde"]
                        
                        factor= self.Model_used[name]["factor"]
                        self.Model_used[name]["w theta"]= [self.Model_used[id_model]["w theta"][i]/factor for i in range(len(self.best_Model_used[id_model]["w theta"]))]
                        self.Model_used[name]["n nx"]= self.best_Model_used[id_model]["n nx"]
            
        