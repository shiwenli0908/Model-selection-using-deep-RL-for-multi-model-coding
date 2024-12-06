# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:10:30 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from Measures import my_bin,my_inv_bin

from Context_Arithmetic import Context_Aritmetic_Encoder,Context_Aritmetic_Decoder
from Measures import get_snr,get_quality,entropy
import pywt 
    

class Khan_Encoder(Context_Aritmetic_Encoder):
    def __init__(self,level,wave_test,M=9,initial_occurrence_first_Khan=[1,1],
                 initial_occurrence_second_Khan=[1,1,1,1],
                 adaptive_Khan=True,verbose_KE=False):
        # Constructeur de la classe Antonini_Encoder
        
        # Initialisation des paramètres spécifiques à Antonini_Encoder
        
        self.initial_occurrence_first_Khan = initial_occurrence_first_Khan # Probabilité initiale équivalente pour la première passe.
        self.initial_occurrence_second_Khan = initial_occurrence_second_Khan  # Probabilité initiale équivalente pour la seconde passe.
        self.verbose_KE = verbose_KE # Mode débogage pour Antonini_Encoder.
        self.adaptive_Khan = adaptive_Khan # 0: le codeur arithmétique n'est pas adaptatif, 1: le codeur arithmétique est adaptatif.
        
        
        
        # ... Autres constantes, initialisations, etc. ...
        self.level=level
        self.wave_test=wave_test
        self.alphabet_first_Khan = ['RR', 'SS'] # Alphabet de la première passe : 'R' (0 non significatif), 'S' (1 significatif).
        self.alphabet_second_Khan = ['-', '+','Z','R']  # Alphabet de la seconde passe : '+' '-' (polarité), 'z' indique que le coefficient n'est pas significatif mais que un de ces enfant l'est.
        
        # Initialisation du codeur arithmétique de contexte adaptatif en appelant le constructeur de la classe parente.
        #super().__init__(M)
        Context_Aritmetic_Encoder.__init__(self,M)
        # Sauvegarde des valeurs courantes de l'état du codeur arithmétique.
        self.memoire_l = self.l
        self.memoire_h = self.h
        self.memoire_follow = self.follow
         
        
    def get_x_rec_Khan(self,coefs):
        x=[np.array(coefs[:int(self.N/2**self.level)])*self.level]
        start=int(self.N/2**self.level)
        for i in range(self.level ):
            # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
            end=2*start
    
            x.append(np.array(coefs[start:end])*self.level)
            start=end
    
        return pywt.waverec(x,self.wave_test, mode='periodization')
        
        
            
    
    def reset_Khan_Encoder(self,coefs,metric,quality,bmax):
        # Réinitialisation des paramètres pour commencer un nouvel encodage
        
        self.reset_Context_Aritmetic_Encoder()  # Réinitialisation des variables de la classe Context_Aritmetic_Encoder.

        
        self.N = len(coefs)  # Détermination de la longueur du vecteur à coder.
        #self.x_test=self.get_x_rec_Khan(coefs)
        self.code = [] # Initialisation de la suite binaire qui sera retournée en tant que résultat du codage.
        self.flag = np.zeros(self.N) # Drapeau indiquant si un coefficient a déjà été considéré significatif. 0 : jamais significatif, 1 : au moins une fois significatif.
        self.coefs = list(coefs) # Initialisation de la liste des coefficients à coder.
        self.res = list(self.coefs) # Initialisation du résidu qui sera mis à jour après le codage d'un coefficient significatif.
        self.coefs_rec = np.zeros(self.N) # Initialisation des coefficients reconstruits.
        self.threshold = 0.5 # Initialisation du seuil à 0.5. # Étant donné que les coefficients sont compris entre -1 et 1, le seuil est initialisé à 0.5.
        self.symbol = []  # Liste pour stocker les symboles codés.
        self.occurrence_first_Khan = list(self.initial_occurrence_first_Khan)  # Initialisation de l'occurrence pour la première passe, qui peut être adaptée au fur et à mesure du codage.
        self.occurrence_second_Khan = list(self.initial_occurrence_second_Khan)  # Initialisation de l'occurrence pour la seconde passe, qui peut être adaptée au fur et à mesure du codage.
        self.cumulate_occurrence_first_Khan = list(accumulate(self.initial_occurrence_first_Khan)) # Calcul des cumuls des occurrences pour la première passe et la seconde passe.
        self.cumulate_occurrence_second_Khan = list(accumulate(self.initial_occurrence_second_Khan))
        self.nb_coefs = 0  # Nombre de coefficients significatif codé
        self.quality_K = quality # Contrainte de qualité.
        self.bmax = bmax # débit max
        self.metric=metric  # m"trique utilisée
        self.occurrence_first_true_Khan = list(self.initial_occurrence_first_Khan) # Occurrence réelle d'apparition des symboles pour la première passe.
        self.occurrence_second_true_Khan = list(self.initial_occurrence_second_Khan) # Occurrence réelle d'apparition des symboles pour la seconde passe.
    
    
        ###### ici sont définis le nombre de coefficients maximale codable dépendnant de br
        nb_coefs_max=bmax#self.N*8#self.N*(self.quality_K-1.76)/(6.02)#br#
        self.nb_bits_coefs_max=int(np.ceil(np.log2(np.max([1,nb_coefs_max]))))
        self.nb_coefs_max=2**self.nb_bits_coefs_max-1
        #print("EZW, br={}, nb_bits_coefs_max={}, nb_coefs_max={}".format(br,self.nb_bits_coefs_max,self.nb_coefs_max))
        self.flag = np.zeros(self.N) # Drapeau indiquant si un coefficient a déjà été considéré significatif. 0 : jamais significatif, 1 : au moins une fois significatif.
        
        ##### liste de même taille que coefs indiquant le numéro de bande pour chaque coefficient
        self.list_bande=np.zeros(self.N,dtype=int)
        start_bande=int(self.N/2)
        end_bande=self.N
        for b in range(self.level):
    
            self.list_bande[start_bande:end_bande]=self.level-b
            end_bande=start_bande
            start_bande=int(start_bande/2)
        
        #print("list bande",self.list_bande)
    
    
        self.flag_tree = np.zeros(self.N) # Drapeau indiquant si un coefficient enfant doit étre codé
    
    
        if self.metric=="MSE":
            self.cst=self.level**2
        elif self.metric=="RMSE":
            self.cst=self.level
        elif self.metric=="SNR":
            x_test=self.get_x_rec_Khan(coefs)
            MSE_x=get_quality(x_test,np.zeros(self.N), "MSE")
            
            self.quality_K=MSE_x/10**(-quality/10)
            self.metric="MSE"
            self.cst=self.level**2
        
    
    
    def get_ind_childs(self,ind_coef):
        """
        Foction qui donne tous les enfants de ind_coefs
        cette focntion dépend de N et du nombre de sous bande
        """
        
        childs=[]

        level_ind_coef=self.list_bande[ind_coef]
        if level_ind_coef<self.level:
            #print("level_ind_coef",level_ind_coef)
            if level_ind_coef==0:
                ind_coef +=int(self.N/(2**self.level))
                childs.extend([ind_coef])
                level_ind_coef+=1
            
            for k in range(level_ind_coef,self.level):
        
                childs.extend([u for u in range(ind_coef*2**(k-level_ind_coef+1),(ind_coef+1)*2**(k-level_ind_coef+1))])
                
        return childs        
        
    

        
        
    def get_symbol_first_Khan(self):
        """
        Fonction qui encode les coefficients de la première passe pour le plan de bits courant
        

        Returns 1 ou 0 si la longueur du mot de code servant à coder le résidu devient suppérieur à la contrainte br
        -------
        int
            DESCRIPTION.

        """
  
        for i in range(self.N):# on balaye tous les flags
            if self.flag[i]!=0:
                if self.res[i]>=0:
                       
                    x=1 ## corespond à l'indice de "S"  dans le dictionnaire dans le dictionnaire de la première passe
                    #print("x S",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    
                    code_first=self.encode_one_symbol(x,self.occurrence_first_Khan,self.cumulate_occurrence_first_Khan)
                    
                    
                    

                    #### mise à jours des variables
                    if len(self.code)+len(code_first)+self.follow+2<=self.bmax-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                        self.symbol.append("SS") # on ajoute le symbole à la liste
                        self.coefs_rec[i]+=self.threshold/2# mise à jour du coef rec
                        self.res[i]-=self.threshold/2
                        self.code.extend(code_first)
                        
                        if self.adaptive_Khan:
                            self.occurrence_first_Khan[x]+=1 # mise à jours du dictionnaire si la contrainte de débit est respécté
                            #print("self.cumulate_occurrence_first av S",self.cumulate_occurrence_first)
                            self.cumulate_occurrence_first_Khan[x]+=1 # mise à jour de cumulate occurrence
                            #print("self.cumulate_occurrence_first ap S",self.cumulate_occurrence_first)
                        
                        self.occurrence_first_true_Khan[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                            
                        self.nb_coefs+=1
                        
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow  
                        
                        
                        #x_rec=self.get_x_rec_Khan(self.coefs_rec)
                        q=self.cst*get_quality(self.coefs,self.coefs_rec,self.metric)
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(self.x_test,lw=2,label='self coefs')
                        plt.plot(x_rec,lw=2,label='self coefs_rec, {}={:.2f}'.format(self.metric,q))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        """
                        #if self.metric=="SNR" or self.metric=="SNR_L1":
                        #    if q>=self.quality_K :
                        #        return 0 
                        #else :
                        if q<=self.quality_K and self.nb_coefs+1<=self.nb_coefs_max:
                            return 0 
                    else :
                        return 0

                        
                else :
                    x=0 # corresond à l'indice "R" dans le dictionaire de la première passe
                    #print("x R",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    code_first=self.encode_one_symbol(x,self.occurrence_first_Khan,self.cumulate_occurrence_first_Khan)
                    #print("x",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    
                    
                        

                    #### mise à jours des variables 
                    if len(self.code)+len(code_first)+self.follow+2<=self.bmax-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                        self.symbol.append("RR") # on ajoute le symbole à la liste
                        self.coefs_rec[i]-=self.threshold/2# mise à jour du coef rec
                        self.res[i]+=self.threshold/2
                        self.code.extend(code_first)
                        
                        
                        if self.adaptive_Khan:
                            self.occurrence_first_Khan[x]+=1 # mise à jours du dictionnaire si la contrainte de débit est respécté
                            self.cumulate_occurrence_first_Khan[0]+=1 # mise à jour de cumulate occurrence
                            self.cumulate_occurrence_first_Khan[1]+=1 # mise à jour de cumulate occurrence
                            
                        #print("self.cumulate_occurrence_first ap R",self.cumulate_occurrence_first)
                        self.occurrence_first_true_Khan[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                        
                        
                        self.nb_coefs+=1
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow 
                        
                        
                        
                        
                        #x_rec=self.get_x_rec_Khan(self.coefs_rec)
                        q=self.cst*get_quality(self.coefs,self.coefs_rec,self.metric)
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(self.x_test,lw=2,label='self coefs')
                        plt.plot(x_rec,lw=2,label='self coefs_rec, {}={:.2f}'.format(self.metric,q))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        """
                        #if self.metric=="SNR" or self.metric=="SNR_L1":
                        #    if q>=self.quality_K and self.nb_coefs+1<=self.nb_coefs_max :
                        #        return 0 
                        #else :
                        if q<=self.quality_K and self.nb_coefs+1<=self.nb_coefs_max :
                            return 0 
                        
                    else:
                        return 0

        
        return 1    # le plan de bits a entièrement été codé  
        
        
    
    def get_symbol_second_Khan(self):
        """
        Fonction qui encode les coefficients de la seconde passe pour le plan de bits courant
        
        -------
        return bool
            DReturns 1 ou 0 si la longueur du mot de code servant à coder le résidu devient suppérieur à la contrainte br

        """
        
        self.flag_tree=np.zeros(self.N) # réinitialisation de self_tree à chaque pass
        
        occurrence_second_Khan=list(self.occurrence_second_Khan)
        cumulate_occurrence_second_Khan=list(self.cumulate_occurrence_second_Khan)
        occurrence_second_true_Khan=list(self.occurrence_second_true_Khan)
        
        
        
        code_ZR=[] # initialisation de la liste de bits servant à représneter les symboles non significatif 
        symbol_ZR=[]
        for i in range(self.N):# on balaye tous les flags
            if self.flag[i]==0 and self.flag_tree[i]==0: ### le coefficient n'a jamais été significatif
                 #### le coefficient appartient à la première bande
            
                if np.abs(self.res[i])>=self.threshold: # le symbole est significatif on encode un + ou un moins -, il faudra alors tester ces enfants si il n'ont jamais été significatifs

                    x=int((np.sign(self.res[i])+1)/2) ## corespond à l'indice de "-": 0 ou "+" : 1  dans le dictionnaire de la seconde passe
                    #print("x",x)
                    
                    code_second=self.encode_one_symbol(x,occurrence_second_Khan,cumulate_occurrence_second_Khan)
                    #print("code_second",code_second)
                    #print("code_ZR",code_ZR)
                    #print("occurrence_second",occurrence_second)
                    #print("cumulate_occurrence_second",cumulate_occurrence_second)

                    #print("len(code)",len(self.code)+len(code_ZR)+len(code_second)+self.follow+2)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    #### mise à jours des variables 
                    if len(self.code)+len(code_ZR)+len(code_second)+self.follow+2<=self.bmax-self.nb_bits_coefs_max and self.nb_coefs+1+len(symbol_ZR)<=self.nb_coefs_max :
                        self.flag[i]=np.sign(self.res[i]) 
                        
                        ######### on indique dans flag_tree qu'il faudra tester les deux enfants de coef[i] si l'on ne se trouve pas dans la dernière sous bande
                        #if self.list_band(i)<self.level:
                        #    print("get_ind_child(i)",self.get_ind_child(i))
                        #    self.flag_tree[self.get_ind_child(i)]=0
                        #print("flag_tree",self.flag_tree)
                        
                        self.symbol.extend(symbol_ZR) # on ajoute les R et Z
                        if self.flag[i]>0:
                            self.symbol.append("+") # on ajoute le symbole à la liste
                        else :
                            self.symbol.append("-") # on ajoute le symbole à la liste
                        self.coefs_rec[i]+=self.flag[i]*(self.threshold+self.threshold/2)# mise à jour du coef rec
                        self.res[i]-=self.flag[i]*(self.threshold+self.threshold/2)
                        
                        self.code.extend(code_ZR)
                        self.code.extend(code_second)
                        #print("len(code)2",len(self.code)+self.follow+2)
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(self.coefs,lw=2,label='coefs')
                        plt.plot(self.coefs_rec,lw=2,label='coefs+1')
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        print("symbol",self.symbol)
                        """
                        

                        if self.adaptive_Khan:
                            occurrence_second_Khan[x]+=1 
                            for xx in range(x,4):
                                cumulate_occurrence_second_Khan[xx]+=1 # mise à jour de cumulate occurrence
                        occurrence_second_true_Khan[x]+=1
                        
                        
                        self.occurrence_second_Khan=occurrence_second_Khan # mise à jours du dictionnaire si la contrainte de débit est respécté
                        self.cumulate_occurrence_second_Khan=cumulate_occurrence_second_Khan  # mise à jour de cumulate occurrence
                        self.occurrence_second_true_Khan=occurrence_second_true_Khan# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                        
                    
    
                        self.nb_coefs+=1
                        self.nb_coefs+=len(symbol_ZR)
                        
                        code_ZR=[]
                        symbol_ZR=[]
                        
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow  
                            
                        
                            
                        
                        
                        
                        
                        
                        #x_rec=self.get_x_rec_Khan(self.coefs_rec)
                        q=self.cst*get_quality(self.coefs,self.coefs_rec,self.metric)
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(self.x_test,lw=2,label='self coefs')
                        plt.plot(x_rec,lw=2,label='self coefs_rec, {}={:.2f}'.format(self.metric,q))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        """
                        #if self.metric=="SNR" or self.metric=="SNR_L1":
                        #    if q>=self.quality_K:
                        #        return 0 
                        #else :
                        if q<=self.quality_K and self.nb_coefs+1<=self.nb_coefs_max:
                            return 0   
                    else :
                        return 0
                    
                else : 
                    ####### le coefficient n'est pas significatif il faut vérifier que ces enfants n'ont jamais été significatif ne sont pas significatif
                    """
                    #tester si les enfants de coefs[i] sont significatif 
                    """
                    childs=self.get_ind_childs(i)
                    #print("i",i,childs)
                    test=0
                    for k in range(len(childs)):
                        if np.abs(self.res[childs[k]])>=self.threshold:
                            
                            ##### encodage de z 
                            x=2 # corespond à Z
                            code_ZR.extend(self.encode_one_symbol(x,occurrence_second_Khan,cumulate_occurrence_second_Khan))
                            
                            if self.adaptive_Khan :
                                occurrence_second_Khan[x]+=1
                                for kk in range(x,4):
                                    cumulate_occurrence_second_Khan[kk]+=1
                            occurrence_second_true_Khan[x]+=1
                            
                            symbol_ZR.append('Z')
                            test=1
                            break
                        test=0
                        
                    #print("test",test)
                    if test==0:
                        
                        ##### encodage de R
                        x=3 # corespond à R
                        code_ZR.extend(self.encode_one_symbol(x,occurrence_second_Khan,cumulate_occurrence_second_Khan))
                        
                        
                        if self.adaptive_Khan :
                            occurrence_second_Khan[x]+=1
                            for kk in range(x,4):
                                cumulate_occurrence_second_Khan[kk]+=1
                        occurrence_second_true_Khan[x]+=1
                        symbol_ZR.append('R')
                        
                        self.flag_tree[childs]=1 # on indique dans flag_tree que coef[i] n'a plus de déscendant il ne seront plus évalué
                    #print("self.flag_tree",self.flag_tree)
   
    
        
            
        #### on encode la fin de la passe avant d'aller à la suivante

        
       
        #print("fin du plan de bits")
        #print("len(code)",len(self.code)+len(code_ZR)+self.follow+2)
        if len(self.code)+len(code_ZR)+self.follow+2<=self.bmax-self.nb_bits_coefs_max and self.nb_coefs+len(code_ZR)<=self.nb_coefs_max :
            self.code.extend(code_ZR)
            
            self.occurrence_second_Khan=occurrence_second_Khan # mise à jours du dictionnaire si la contrainte de débit est respécté
            self.cumulate_occurrence_second_Khan=cumulate_occurrence_second_Khan  # mise à jour de cumulate occurrence
            self.occurrence_second_true_Khan=occurrence_second_true_Khan# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
            
    
            #print("code_ZR",code_ZR)
            #print("occurrence_second",self.occurrence_second)
            #print("cumulate_occurrence_second",self.cumulate_occurrence_second)
            
            
            self.symbol.extend(symbol_ZR)
            
            
    
            self.nb_coefs+=len(symbol_ZR)
            
            self.memoire_l=self.l
            self.memoire_h=self.h
            self.memoire_follow=self.follow  
    
      
    
            self.threshold/=2
    
            return 1 # le plan de bits à entièrement été codé 
        else :
            return 0
             

      

    def get_code_res_Khan(self,coefs,metric,quality,bmax):
        # Obtenir le code résultant et le nombre de bits par coefficient
        self.reset_Khan_Encoder(coefs,metric,quality,bmax)
        
        if self.nb_coefs_max==0:
            return []
        
        
        while self.get_symbol_first_Khan() and self.get_symbol_second_Khan():
            pass
        """
        true=1
        while true==1:
            true=self.get_symbol_first()
            print("len(code)",len(self.code),"br",br,true)
            
            if true==1:
                true= self.get_symbol_second()
                print("len(code)",len(self.code),"br",br,true)
        """
        code_end=self.finish(self.memoire_l,self.memoire_follow)
        
        code_nb_coefs=my_bin(self.nb_coefs,self.nb_bits_coefs_max) # on code la longueur
        #print("code_nb_coefs",code_nb_coefs)
        self.code.extend(code_end)
        
        
        ###### on place le train binaire en tête indiquant combien de coefficients ont été encodé
        
        #print(code_nb_coefs)
        #print(self.code)
        #print(code_nb_coefs+self.code)
        #print("symbol",self.symbol)
        return code_nb_coefs+self.code
        #return self.code











class Khan_Decoder(Context_Aritmetic_Decoder):
    def __init__(self,level,wave_test,N=128,M=9,initial_occurrence_first_Khan=[1,1],
                 initial_occurrence_second_Khan=[1,1,1,1],
                 adaptive_Khan=True,verbose_KD=False):
        
        # Constructeur de la classe Antonini_Decoder
        
        # Initialisation des paramètres spécifiques à Antonini_Decoder
        self.N=N #Longueur du vecteur à coder.
        self.initial_occurrence_first_Khan = initial_occurrence_first_Khan # Probabilité initiale équivalente pour la première passe.
        self.initial_occurrence_second_Khan = initial_occurrence_second_Khan # Probabilité initiale équivalente pour la seconde passe.
        self.verbose_KD = verbose_KD # Mode débogage pour Antonini_Decoder.
        self.adaptive_Khan = adaptive_Khan  # 0: le codeur arithmétique n'est pas adaptatif, 1: le codeur arithmétique est adaptatif.
 
        
        # ... Autres constantes, initialisations, etc. ...
        self.level=level
        self.wave_test=wave_test
        self.alphabet_first_Khan = ['RR', 'SS']  # Alphabet de la première passe : 'R' (0 non significatif), 'S' (1 significatif).
        self.alphabet_second_Khan = ['-', '+','Z','R'] # Alphabet de la seconde passe : '0' (0), '1' (1), '+' (longueur de séquence), '-' (polarité), 'E' (fin).
        
        # Initialisation du codeur arithmétique de contexte adaptatif en appelant le constructeur de la classe parente.
        
        
        Context_Aritmetic_Decoder.__init__(self,M)
        #super().__init__(M)
     
            
    def get_x_rec_Khan(self,coefs):
        x=[np.array(coefs[:int(self.N/2**self.level)])*self.level]
        start=int(self.N/2**self.level)
        for i in range(self.level ):
            # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
            end=2*start
    
            x.append(np.array(coefs[start:end])*self.level)
            start=end
    
        return pywt.waverec(x,self.wave_test, mode='periodization')
        
        
    
    def reset_Khan_Decoder(self,code,bmax):
        # Réinitialisation des paramètres pour commencer un nouvel encodage

        self.reset_Context_Aritmetic_Decoder() # Réinitialisation des variables de la classe Context_Aritmetic_Encoder.
        self.flag=np.zeros(self.N) # drapeau indiquant si le coefficient a déja été significatif. 0 : jamais significatif, 1 : au moins une fois significatif
        self.coefs_rec=np.zeros(self.N) #Initialisation du seuil à 0.5.
        self.threshold=0.5 # les coefficicients sont compris entre -1 et 1 donc le threshold est initialisé à 0.5. Étant donné que les coefficients sont compris entre -1 et 1, le seuil est initialisé à 0.5.       
        self.symbol=[] # Liste pour stocker les symboles décodés.
        self.occurrence_first_Khan=list(self.initial_occurrence_first_Khan) # Initialisation de l'occurrence pour la première passe, qui peut être adaptée au fur et à mesure du codage.
        self.occurrence_second_Khan=list(self.initial_occurrence_second_Khan) # Initialisation de l'occurrence pour la seconde passe, qui peut être adaptée au fur et à mesure du codage.
        self.cumulate_occurrence_first_Khan=list(accumulate(self.initial_occurrence_first_Khan)) # Calcul des cumuls des occurrences pour la première passe et la seconde passe.
        self.cumulate_occurrence_second_Khan=list(accumulate(self.initial_occurrence_second_Khan)) # Calcul des cumuls des occurrences pour la seconde passe et la seconde passe.
        
        
        
        
        nb_coefs_max=bmax#self.N*8
        self.nb_bits_coefs_max=int(np.ceil(np.log2(np.max([1,nb_coefs_max]))))
        self.nb_coefs_max=2**self.nb_bits_coefs_max-1
        
        #print("self.nb_coefs_max",self.nb_coefs_max)
        #print("br",br)
        #print("self.nb_coefs_max",self.nb_coefs_max)
        #print("self.nb_bits_coefs_max",self.nb_bits_coefs_max)
        
        
        
        self.nb_coefs=my_inv_bin(code[0:self.nb_bits_coefs_max])
        
        #print("self.nb_coefs dec",self.nb_coefs)
        self.nb_coefs_dec=0
        
        
        
        self.code=code[self.nb_bits_coefs_max:] # Détermination de la longueur du vecteur à coder.
        #print("self.code dec",self.code)
    
        ##### liste de même taille que coefs indiquant le numéro de bande pour chaque coefficient
        self.list_bande=np.zeros(self.N,dtype=int)
        start_bande=int(self.N/2)
        end_bande=self.N
        for b in range(self.level):
    
            self.list_bande[start_bande:end_bande]=self.level-b
            end_bande=start_bande
            start_bande=int(start_bande/2)
        
        #print("list bande",self.list_bande)
    
    
        self.flag_tree = np.zeros(self.N) # Drapeau indiquant si un coefficient enfant doit étre codé
    
    
    
    
    
    def get_ind_childs(self,ind_coef):
        """
        Foction qui donne tous les enfants de ind_coefs
        cette focntion dépend de N et du nombre de sous bande
        """
        
        childs=[]

        level_ind_coef=self.list_bande[ind_coef]
        if level_ind_coef<self.level:
            #print("level_ind_coef",level_ind_coef)
            if level_ind_coef==0:
                ind_coef +=int(self.N/(2**self.level))
                childs.extend([ind_coef])
                level_ind_coef+=1
            
            for k in range(level_ind_coef,self.level):
        
                childs.extend([u for u in range(ind_coef*2**(k-level_ind_coef+1),(ind_coef+1)*2**(k-level_ind_coef+1))])
                
        return childs        
            
    
    
    
    
    
    
    
    
    
    def get_symbol_first_Khan(self):
        # Décode un symbole de la première passe
        

        for i in range(self.N):
            if (self.flag[i]!=0):
                symbol_first=self.decode_one_symbol(self.code,self.alphabet_first_Khan,self.occurrence_first_Khan,self.cumulate_occurrence_first_Khan)
                self.symbol.append(symbol_first)
                self.nb_coefs_dec+=1
                
                if symbol_first=="SS":
                    x=1
                    if self.adaptive_Khan:
                        self.cumulate_occurrence_first_Khan[1]+=1
                    self.coefs_rec[i]+=self.threshold/2
                else:
                    x=0
                    if self.adaptive_Khan:
                        self.cumulate_occurrence_first_Khan[0]+=1
                        self.cumulate_occurrence_first_Khan[1]+=1
                    self.coefs_rec[i]-=self.threshold/2
                    
                    
                    
                if self.adaptive_Khan:    
                    self.occurrence_first_Khan[x]+=1
                
                if self.nb_coefs_dec==self.nb_coefs:
                    
                    return 0
        return 1
                
                    

        
 
        
    def get_symbol_second_Khan(self):
        # Décode un symbole de la seconde passe 
        
        
        
        pointeur=0 # pointeur indiquant la position du coefficients courant dans le vecteur coefs 
               
        
        
        while pointeur<self.N:
            if self.flag[pointeur] ==0 and self.flag_tree[pointeur]==0:
                #print(pointeur)
                ### on décode un symbole
                symbol_second=self.decode_one_symbol(self.code,self.alphabet_second_Khan,
                                self.occurrence_second_Khan,self.cumulate_occurrence_second_Khan)
             
                x=self.alphabet_second_Khan.index(symbol_second)
                
                if self.adaptive_Khan:
                    self.occurrence_second_Khan[x]+=1
                    for xx in range(x,4):
                        self.cumulate_occurrence_second_Khan[xx]+=1
    
                self.symbol.append(symbol_second)
                
                #print("symbol_second",symbol_second)
        
                    
    
                if symbol_second == '+':
                    self.coefs_rec[pointeur]+=self.threshold+self.threshold/2
                    self.flag[pointeur]=1
    
                elif symbol_second == '-':
                    self.coefs_rec[pointeur]-=self.threshold+self.threshold/2
                    self.flag[pointeur]=-1
                
                elif symbol_second =="R":
                    childs=self.get_ind_childs(pointeur)
                    self.flag_tree[childs]=1
                elif symbol_second =="Z":
                    pass
                
                
                pointeur+=1
                self.nb_coefs_dec+=1
                if self.nb_coefs_dec==self.nb_coefs:
                    return 0
            else :
                pointeur+=1    

    
        self.threshold/=2
        self.flag_tree=np.zeros(self.N)
        #print("symbol",self.symbol)
        return 1
    
    
    def get_coefs_rec_Khan(self,code,bmax):
        # Obtenir les coefficients reconstruit à partir de la suite binaire code
        
       
        
        self.reset_Khan_Decoder(code,bmax)
        self.ini_codeword(self.code)
        
       
            
        
        while self.nb_coefs_dec<self.nb_coefs:
            #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
            #print("count",self.count,"len(code)",len(code))
            self.get_symbol_first_Khan()
            if self.nb_coefs_dec<self.nb_coefs:
                #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
                #print("count",self.count,"len(code)",len(code))
                self.get_symbol_second_Khan()

            #pass
        #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
        #print("count",self.count,"len(code)",len(code))
        
      
        return self.coefs_rec






# Programme principal
if __name__ == "__main__":

    from Models import Model_poly
    from Normalize import normalize
    import cProfile
    
    

    metric="SNR"
    quality=-15
    unity= "dB"
    bmax=1000#bits
    M=9
    
    adaptive_Khan =True
    verbose = False
    

    initial_occurrence_first_Khan=[1,1]
    initial_occurrence_second_Khan=[1,1,1,1]
                         
    N=128
    fn=50
    fs=6400
    
    t=np.linspace(0,(N-1)/fs,N)
    
 
    
    
    sigma=0.2 # écart type du bruit introduit dans le signal test
    

    ###############  test polynôme d'ordre k
    order=8
    theta=np.random.uniform(-0.2,0.2,order+1) #[-0.4, -0.3,  0.2 , 0.05 , 0.8 ,-0.3]#
   
    model_poly=Model_poly(fn,fs,N,verbose)
  
    x_test=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    x_test,_=normalize(x_test)

    #x_test=np.zeros(N)
    #x_test[N-1]=0.687
    #x_test[N-2]=-0.4
    
    #x_test=idct(x_test)/2
    
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
    
    #wav_test=pywt.Wavelet('coif5')
    #level=2#int(np.ceil(np.log2(N)))
    
    
    
    # Exemple d'utilisation
 
    wave_test = 'coif5'
    level = 7
    mode='periodization'
    
    KE=Khan_Encoder(level,wave_test,M,initial_occurrence_first_Khan,initial_occurrence_second_Khan,adaptive_Khan,verbose_KE=verbose)


    # Créez un objet cProfile
    profile = cProfile.Profile()
    
    # Démarrez le profiler avant d'appeler votre fonction principale
    profile.enable()
    coefs = pywt.wavedec(x_test, wave_test, mode=mode, level=level)
    
    # Initialiser une liste pour stocker les coefficients avec la forme de x_test
    coefs_L = []
    
    # Pour chaque niveau de décomposition
    for i in range(level + 1):
        # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
        coefs_L.extend(coefs[i]/level)






   

    for k in range(1):
        code=KE.get_code_res_Khan(coefs_L,metric,quality,bmax)
    
    coefs_L_rec_E=KE.coefs_rec
    


    x_rec_E =KE.get_x_rec_Khan(coefs_L_rec_E)
    
    
    # Arrêtez le profiler
    profile.disable()
    
    
    # Affichez les statistiques de profil
    profile.print_stats(sort='cumulative')    
    

    print("Nb bits used / Nb bits max = {} / {} bits".format(len(code),bmax),"{} = {} / {} {}".
          format(metric,get_quality(x_test,x_rec_E,metric),quality,unity))
    print("Nb sym codé / Nb sym max = {} / {}".format(KE.nb_coefs,KE.nb_coefs_max))
    
  

    occurrence_first_Khan_=KE.occurrence_first_Khan
    occurrence_second_Khan_=KE.occurrence_second_Khan
    print("Occurrence des symboles des premières passe",occurrence_first_Khan_)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second_Khan_)
        
        
    occurrence_first_Khan=np.array(KE.occurrence_first_true_Khan)-1
    occurrence_second_Khan=np.array(KE.occurrence_second_true_Khan)-1
    
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
    

    
    
    
    
    
    
    

    #################### décodeur

    KD=Khan_Decoder(level,wave_test,N,M,initial_occurrence_first_Khan,initial_occurrence_second_Khan,adaptive_Khan,verbose_KD=verbose)

    


        
    coefs_L_rec_D=KD.get_coefs_rec_Khan(code,bmax)
    


    x_rec_D =KE.get_x_rec_Khan(coefs_L_rec_D) 
    

    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(coefs_L,lw=2,label='coefs')
    plt.plot(coefs_L_rec_D,lw=2,label='coefs rec')
    plt.xlabel('ind')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Nb bits used / nb bits max = {} / {}, H = {:.1f} bits'.
             format(len(code),bmax,H_tot))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x_test')
    plt.plot(t,x_rec_E,lw=2,label='x_rec_Enc, Nb bits used / nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.
             format(len(code),bmax,metric,metric,get_quality(x_test,x_rec_E,metric),quality,unity))
    plt.plot(t,x_rec_D,lw=2,label='x_rec_Dec, Nb bits used / nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.
             format(len(code),bmax,metric,metric,get_quality(x_test,x_rec_D,metric),quality,unity))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
  