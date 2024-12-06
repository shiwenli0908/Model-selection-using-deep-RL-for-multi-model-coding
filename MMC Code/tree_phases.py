# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:36:56 2023

@author: presvotscor
"""

import numpy as np
from Normalize import normalize



#from  codage_model import Model_Encoder,Model_Decoder
#from  codage_residu import Residual_Encoder,Residual_Decoder
#from  Allocation_two_stages import Allocation_sin_bx_br,Allocation_poly_bx_br,Allocation_pred_samples_bx_br

from Quantization import Quantizer


import matplotlib.pyplot as plt

width=6


verbose = False

N=128 # size of window
fn=50 # nominal frequency
fs=6400 # samples frequency






   
"""
# Open the data
"""

from get_test_signal import get_RTE_signal,get_EPRI_signal

#v1,v2,v3,i1,i2,i3=get_RTE_signal()
#N=128
#fn=50
#fs=6400

#21834 21845 21852 21854 21860 21861 21862 21863 21869 21881 {34, 45, 52, 54, 60, 61, 62, 63, 69, 81}

v1,v3,v2,i1,i2,i3=get_EPRI_signal(21833)

v1/=1
v2/=1
v3/=1

N=256
fn=60
fs=15384.6 


def transform_tree(v1, v2, v3,A,angles,transform):
    v1_=np.copy(v1)
    v2_=np.copy(v2)
    v3_=np.copy(v3)
    if transform=="clarke" :
        
    
        # Calcul des composantes alpha, beta et zéro séquence
        alpha = (2/3) * (v1_ - (1/2) * v2_ - (1/2) * v3_)
        beta = (2/3) * ((np.sqrt(3)/2) * v2_ - (np.sqrt(3)/2) * v3_)
        o= (1/3) * (v1_ + v2_ + v3_)
        return alpha, beta, o

    if transform=="concordia" :
        # Calcul des composantes alpha, beta et zéro séquence
        alpha = np.sqrt((2/3)) * (v1_ - (1/2) * v2_ - (1/2) * v3_)
        beta = np.sqrt((2/3)) * ((np.sqrt(3)/2) * v2_ - (np.sqrt(3)/2) * v3_)
        o= np.sqrt((1/3)) * (v1_ + v2_ + v3_)
        return alpha, beta, o
    if transform=="dqo":
        
        alpha =np.sqrt(2/3)*(v1_*np.cos(angles) +v2_*np.cos(angles-2*np.pi/3)  + v3_*np.cos(angles+2*np.pi/3))-np.sqrt(3/2)*A 
        beta = np.sqrt(2/3)*(-v1_*np.sin(angles) -v2_*np.sin(angles-2*np.pi/3)  - v3_*np.sin(angles+2*np.pi/3)   ) 
        o=np.sqrt(2/3)*((v1_+v2_+v3_)*np.sqrt(2)/2)
        return alpha, beta, o
    return v1_,v2_,v3_
            
    



def inverse_transform_tree(alpha, beta, o,A,angles,transform):
    alpha_=np.copy(alpha)
    beta_=np.copy(beta)
    o_=np.copy(o)
    if transform=="clarke" :    
        # Calcul des phases v1, v2 et v3
        v1 = alpha_+o_
        v2 = - (1/2)*alpha_+(np.sqrt(3)/2) * beta_ +o_
        v3 = - (1/2) * alpha_-(np.sqrt(3)/2) * beta_ +o_
        return v1, v2, v3
        
    if transform=="concordia":
        v1 = np.sqrt((2/3))*(alpha_+(np.sqrt(2)/2)*o_)
        v2 = np.sqrt((2/3))*(-0.5*alpha_+(np.sqrt(3)/2)*beta_+(np.sqrt(2)/2)*o_)
        v3 = np.sqrt((2/3))*(-0.5*alpha_-(np.sqrt(3)/2)*beta_+(np.sqrt(2)/2)*o_)

        return v1, v2, v3
    
    
    if transform=="dqo":
        alpha_+=np.sqrt(3/2)*A 
        v1 = np.sqrt(2/3)*(alpha_*np.cos(angles) -beta_*np.sin(angles)  + o_*np.sqrt(2)/2 )
        v2 = np.sqrt(2/3)*(alpha_*np.cos(angles-2*np.pi/3) -beta_*np.sin(angles-2*np.pi/3) + o_*np.sqrt(2)/2   ) 
        v3 = np.sqrt(2/3)*(alpha_*np.cos(angles+2*np.pi/3) -beta_*np.sin(angles+2*np.pi/3) + o_*np.sqrt(2)/2   ) 
        return v1, v2, v3
    
    return alpha_, beta_,o_









from scipy.optimize import fsolve


def equation(vars, b_tot,N,Kx,Kr,transform):
    
    b_alpha, b_beta, b_o,lambda_ = vars
    w=np.ones(3)*2**Kx
    if transform=="clarke":
        return [
            (w[0]**2*np.log(2)/12)*3*2**(-2*b_alpha/N) -lambda_,
            (w[1]**2*np.log(2)/12)*3*2**(-2*b_beta/N) -lambda_,
            (w[2]**2*np.log(2)/12)*6*2**(-2*b_o/N) -lambda_,
            b_tot-b_alpha-b_beta-b_o
        ]
    if transform=="concordia":
        return [
            (w[0]**2*np.log(2)/12)*3*2**(-2*b_alpha/N) -lambda_,
            (w[1]**2*np.log(2)/12)*3*2**(-2*b_beta/N) -lambda_,
            (w[2]**2*np.log(2)/12)*3*2**(-2*b_o/N) -lambda_,
            b_tot-b_alpha-b_beta-b_o
        ]
    if transform=="dqo":
        return [
                    -(2/3)*2**(2*(Kx[0]-Kr[0]))*np.log(2)*2**(-2*b_alpha/N) -lambda_,
                    -(2/3)*2**(2*(Kx[1]-Kr[1]))*np.log(2)*2**(-2*b_beta/N) -lambda_,
                    -(2/3)*2**(2*(Kx[2]-Kr[2]))*np.log(2)*2**(-2*b_o/N) -lambda_,
                    b_tot-b_alpha-b_beta-b_o
                ]







def allocation_optimal(b_tot,N,Kx,Kr,transform):
    if transform=='none':
        return [b_tot/3, b_tot/3, b_tot/3]

    # Supposons une valeur initiale pour les variables b_alpha, b_beta, et b_gamma
    initial_guess = [b_tot/3, b_tot/3, b_tot/3,5]
    
    # Utilisation de fsolve pour résoudre les équations
    result = fsolve(equation, initial_guess, args=(b_tot,N,Kx,Kr,transform,))
    b_alpha, b_beta, b_o ,lambda_= result
    #print("result=[{:.1f},{:.1f},{:.1f},{:.3f}]".format(*result))
    return b_alpha, b_beta, b_o















def est_equilibre(v1, v2, v3,tol):
    # Calcul des valeurs efficaces (RMS) pour chaque tension de phase
    v_rms = np.array([np.sqrt(np.mean(np.square(v))) for v in (v1, v2, v3)])
    
    v_rms/=np.max(v_rms)
    
    
    # Vérification de l'égalité des tensions de ligne (tensions efficaces)
    
    


    for k in range(2):
        if np.abs(v_rms[0]-v_rms[k+1])> tol :
            print("v_rms=[{:.2f},{:.2f},{:.2f}]".format(*v_rms),"False")
            return False
    print("v_rms=[{:.2f},{:.2f},{:.2f}]".format(*v_rms),"True")
    return True











# Programme principal
if __name__ == "__main__":
    from Models import Model_sin
    from Bits_allocation import Allocation
    
    
    q_x=Quantizer(False)
    # Exemple 
    #"""

    nb_w=5
    btot=128
    bm=int(0.2*btot)
    
    transform="dqo"
    x=[v1[0:N*nb_w],v2[0:N*nb_w],v3[0:N*nb_w]]
    
    
    x_transform=np.zeros((3,N*nb_w))
    
    
    theta_tri=np.zeros((3,nb_w,3))
    
    r_transform=np.zeros((3,N*nb_w))
    r_q_transform=np.zeros((3,N*nb_w))
    
    eqr=np.zeros(N*nb_w)
    
    x_rec_transform=np.zeros((3,N*nb_w))
    x_rec=np.zeros((3,N*nb_w))

    
    
  
    model_sin=Model_sin(fn,fs,N,False)
    A=Allocation(verbose=False)
    
    
    
    
    
    t=np.linspace(0,(N-1)/fs,N)
 
    for k in range(nb_w):
        """
        estimation de a,f,phi
        """
        Kx=np.zeros(3)
        a=0
        f=0
        phi=0
        if k>0:
            """
            for phase in range(3):
                _,Kx[phase]=normalize(x[phase][(k-1)*N:k*N])

            
                theta_tri[phase][k]=model_sin.get_theta_sin(x[phase][(k-1)*N:k*N]*2**(-Kx[phase])) 
            """    

            for phase in range(3):
                _,Kx[phase]=normalize(x[phase][(k)*N:(k+1)*N])

            
                theta_tri[phase][k]=model_sin.get_theta_sin(x[phase][(k)*N:(k+1)*N]*2**(-Kx[phase])) 
                

            

            a=(theta_tri[0][k][0]*2**Kx[0]+theta_tri[1][k][0]*2**Kx[1]+theta_tri[0][k][0]*2**Kx[2])/3
            f=(theta_tri[0][k][1]+theta_tri[1][k][1]+theta_tri[2][k][1])/3
            
    
            phi=(theta_tri[0][k][2]+theta_tri[1][k][2]+theta_tri[2][k][2])/3
            
            if np.abs(theta_tri[0][k][2]-phi)>np.pi/2:
                phi+=np.sign(theta_tri[0][k][2]-phi)*2*np.pi/3
                
            #print("theta1={:.4f}, phi={:.4f}".format(theta1[2],phi))
            
     
            
            theta1=[a,f,phi]
            theta2=[a,f,phi-2*np.pi/3]
            theta3=[a,f,phi+2*np.pi/3]
       
            x_model_1=model_sin.get_model_sin(t, *theta1)
            x_model_2=model_sin.get_model_sin(t, *theta2)
            x_model_3=model_sin.get_model_sin(t, *theta3)
            
            """
            plt.figure(figsize=(width,4), dpi=100)
            plt.plot(t,x[0][(k-1)*N:k*N]/1000,lw=1,label='x1')
            plt.plot(t,x_model_1/1000,lw=1,label='theta1=[{:.2f},{:.2f},{:.2f}]'.format(*theta_tri[0][k]))
            plt.plot(t,x[1][(k-1)*N:k*N]/1000,lw=1,label='x2')
            plt.plot(t,x_model_2/1000,lw=1,label='theta2=[{:.2f},{:.2f},{:.2f}]'.format(*theta_tri[1][k]))
            plt.plot(t,x[2][(k-1)*N:k*N]/1000,lw=1,label='x3')
            plt.plot(t,x_model_3/1000,lw=1,label='theta3=[{:.2f},{:.2f},{:.2f}]'.format(*theta_tri[2][k]))
            plt.xlabel('ind sample')
            plt.ylabel('Magnitude x10e3')
            plt.legend()
            plt.title("w={}".format(k))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
            
            
            plt.figure(figsize=(width,4), dpi=100)
            plt.plot(t,(x[0][(k-1)*N:k*N]-x_model_1)/1000,lw=1,label='x1-mo')
            plt.plot(t,(x[1][(k-1)*N:k*N]-x_model_2)/1000,lw=1,label='x2-mo')
            plt.plot(t,(x[2][(k-1)*N:k*N]-x_model_3)/1000,lw=1,label='x3-mo')
            plt.xlabel('ind sample')
            plt.ylabel('Magnitude x10e3')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
            """
        

            
        angles=2*np.pi*f*t+phi
        x_transform[0][k*N:(k+1)*N],x_transform[1][k*N:(k+1)*N],x_transform[2][k*N:(k+1)*N]=transform_tree(x[0][k*N:(k+1)*N],x[1][k*N:(k+1)*N],x[2][k*N:(k+1)*N],a,angles,transform)

    
    
        x_rec[0][k*N:(k+1)*N],x_rec[1][k*N:(k+1)*N],x_rec[2][k*N:(k+1)*N] = inverse_transform_tree(x_transform[0][k*N:(k+1)*N], x_transform[1][k*N:(k+1)*N], x_transform[2][k*N:(k+1)*N],a,angles,transform)
    
        
    
        ###################### test allocation optimale de bits
        Kx=np.zeros(3)
        std_=np.zeros(3)
        std_q=np.zeros(3)
        if k>0:
            

            for phase in range(3):
                r_transform[phase][k*N:(k+1)*N]=x_transform[phase][k*N:(k+1)*N]-x_transform[phase][(k-1)*N:k*N]
                
                
                Kx[phase]=np.log2(np.max(np.abs(r_transform[phase][k*N:(k+1)*N])))
                std_[phase]=np.std( r_transform[phase][k*N:(k+1)*N])
        
        
        
                plt.figure(figsize=(width,4), dpi=100)
                plt.plot( r_transform[phase][k*N:(k+1)*N],lw=1,label='phase: {}, a={:.2f}, std={:.2f}'.format(phase+1,2**Kx[phase],std_[phase]))
                plt.xlabel('ind sample')
                plt.ylabel('Magnitude')
                plt.legend()
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
            
            
            """
            
            alpha_n, Kx[0]=normalize(alpha[k*N:(k+1)*N])
            beta_n, Kx[1]=normalize(beta[k*N:(k+1)*N])
            o_n, Kx[2]=normalize(o[k*N:(k+1)*N])
            
            
            plt.figure(figsize=(width,4), dpi=100)
            plt.plot(alpha_n,lw=1,label='alpha_n')
            plt.plot(alpha[(k-1)*N:k*N]*2**(-Kx[0]),lw=1,label='alpha_n_p')
            plt.xlabel('ind sample')
            plt.ylabel('Magnitude x10e3')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
            
            _, Kr[0]=normalize(alpha_n-alpha[(k-1)*N:k*N]*2**(-Kx[0]))
            _, Kr[1]=normalize(beta_n-beta[(k-1)*N:k*N]*2**(-Kx[1]))
            _, Kr[2]=normalize(o_n-o[(k-1)*N:k*N]*2**(-Kx[2]))
            """
     
        #Kr=np.abs(Kr)
        

        al=allocation_optimal(3*btot-3*bm,N/10,Kx,np.zeros(3),transform)
        al=A.round_allocation(al,3*btot-3*bm)
        
        print("al=[{},{},{}], btot={:.0f}, amplitude_max_r=[{:.2f},{:.2f},{:.2f}], std_r=[{:.2f},{:.2f},{:.2f}],".format(al[0]+bm,al[1]+bm,al[2]+bm,np.sum(al)+3*bm,2**Kx[0],2**Kx[1],2**Kx[2],*std_))
        
        if k>0:
            for phase in range(3):
                b=al
                for i in range(N):
                    ind=q_x.get_ind_u(r_transform[phase][k*N+i],al[phase],2*2**Kx[phase],0)
               
                    r_q_transform[phase][k*N+i]=q_x.get_q_u(ind,al[phase],2*2**Kx[phase],0)
               
            
            
    
                std_q[phase]=np.std(r_transform[phase][k*N:(k+1)*N]- r_q_transform[phase][k*N:(k+1)*N])
        
        
        
                plt.figure(figsize=(width,4), dpi=100)
                plt.plot(r_transform[phase][k*N:(k+1)*N],lw=1,label='phase: {}, a={:.2f}, std={:.2f}'.format(phase+1,2**Kx[phase],std_[phase]))
                plt.plot(r_q_transform[phase][k*N:(k+1)*N],lw=1,label='phase: {}, a={:.2f}, std_q={:.2f}'.format(phase+1,2**Kx[phase],std_q[phase]))
                plt.xlabel('ind sample')
                plt.ylabel('Magnitude')
                plt.legend()
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show() 
                
            
            #print("al=[{},{},{}], btot={:.0f}, kx=[{:.2f},{:.2f},{:.2f}], kr=[{:.0f},{:.0f},{:.0f}], kx-Kr=[{:.0f},{:.0f},{:.0f}]".format(al[0]+bm,al[1]+bm,al[2]+bm,np.sum(al)+3*bm,*Kx,*Kr,*(Kx-Kr)))
            
        
    
    w_min=1
    w_max=10
    
    plt.figure(figsize=(width,4), dpi=100)
    for phase in range(3):
        plt.plot(x_transform[phase][w_min*N:w_max*N]/1000,lw=1,label='x transform phase : {}'.format(phase+1))
    plt.xlabel('ind sample')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    
    plt.figure(figsize=(width,4), dpi=100)
    for phase in range(3):
        plt.plot(x[phase][w_min*N:w_max*N]/1000,lw=1,label='x: {}'.format(phase+1))
        plt.plot(x_rec[phase][w_min*N:w_max*N]/1000,lw=1,label='x_rec')

    plt.xlabel('ind sample')
    plt.ylabel('Magnitude x10e3')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show() 
    
    

    

    



    


    """
    ########################### test tolérance système équilibré
    tol=0.1
    equilibre = est_equilibre(x1, x2, x3,tol)
    if equilibre:
        print("Le réseau triphasé est équilibré.")
    else:
        print("Le réseau triphasé n'est pas équilibré.")
        
    """   
        