# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:04:56 2023

@author: presvotscor
"""
"""
# Ouvrir le fichier en mode lecture
with open('data_test.txt', 'r') as file:
    # Lire le contenu du fichier
    content = file.read()

# Extraire la liste à partir du contenu du fichier
# En supposant que la liste est toujours dans la forme indiquée
start = content.find('[')
end = content.find(']')
if start != -1 and end != -1:
    # Extraire la partie entre crochets et la diviser en éléments individuels
    list_str = content[start + 1:end]
    x_test_RTE = [float(item.strip()) for item in list_str.split(',')]

    # Maintenant, 'elements' contient la liste que vous souhaitez récupérer
    print(x_test_RTE[0:10])
"""

import numpy as np
import matplotlib.pyplot as plt

def get_RTE_signal_new():
    Delta_u=18.31055
    Delta_i=4.314
    loaded = np.load(r'D:\Users\presvotscor\Documents\These_clean\Code_MMC\DATA_S2.npz')    
    DATA_S=loaded['DATA_S2']
    
    
    for number in range(len(DATA_S)):
        
        DATA_S[number][0]*=Delta_u
        DATA_S[number][1]*=Delta_u
        DATA_S[number][2]*=Delta_u
        DATA_S[number][3]*=Delta_i
        DATA_S[number][4]*=Delta_i
        DATA_S[number][5]*=Delta_i
        

        
    return DATA_S
    


def get_RTE_signal():
    Delta_u=18.31055
    Delta_i=4.314
    loaded = np.load(r'D:\Users\presvotscor\Documents\These_clean\Code_MMC\DATA.npz')    
    DATA_S=loaded['DATA_S']
 
    for number in range(len(DATA_S)):
        
        DATA_S[number][0]*=Delta_u
        DATA_S[number][1]*=Delta_u
        DATA_S[number][2]*=Delta_u
        DATA_S[number][3]*=Delta_i
        DATA_S[number][4]*=Delta_i
        DATA_S[number][5]*=Delta_i
    return DATA_S
    
"""   
def get_RTE_signal():
    nom_fichier = "monFichier.txt"
    

    # Listes pour stocker les données
    v1=[]
    v2=[]
    v3=[]
    i1=[]
    i2=[]
    i3=[]
    
    # Lire les données depuis le fichier
    with open(nom_fichier, 'r') as file:
        lignes = file.readlines()
        for ligne in lignes:
            elements = ligne.strip().split(',')  # Divise la ligne en éléments en utilisant la virgule comme délimiteur
            if len(elements) >= 1:
                # Ajoute le premier élément (colonne 1) à la liste
                v1.append(float(elements[0]))
                v2.append(float(elements[1]))
                v3.append(float(elements[2]))
                i1.append(float(elements[3]))
                i2.append(float(elements[4]))
                i3.append(float(elements[5]))
    v1=np.array(v1)/290
    v2=np.array(v2)/290
    v3=np.array(v3)/290
    i1=np.array(i1)/290
    i2=np.array(i2)/290
    i3=np.array(i3)/290

    return v1,v2,v3,i1,i2,i3
"""


def get_EPRI_signal(number):

    nom_fichier = "EPRI/{}.csv".format(str(number))
    # Listes pour stocker les données
    v1=[]
    v2=[]
    v3=[]
    i1=[]
    i2=[]
    i3=[]
    
    file=open(nom_fichier, 'r')
    line = file.readlines()[3: :1]
    
    for element in line :
        v1.append(float(element.split(',')[1]))
        v2.append(float(element.split(',')[2]))
        v3.append(float(element.split(',')[3]))
        i1.append(float(element.split(',')[4]))
        i2.append(float(element.split(',')[5]))
        i3.append(float(element.split(',')[6]))

    v1=np.array(v1)
    v2=np.array(v2)
    v3=np.array(v3)
    i1=np.array(i1)
    i2=np.array(i2)
    i3=np.array(i3)
    
    return v1,v2,v3,i1,i2,i3
    #file = open("dataset/{}.csv".format(str(number)), "r")
    
    
    
    

# Programme principal
if __name__ == "__main__":
    fs=6400 # samples frequency
    
    DATA_S2=get_RTE_signal_new()
    number=10
    v1=DATA_S2[number][0]
    v2=DATA_S2[number][1]
    v3=DATA_S2[number][2]
    i1=DATA_S2[number][0]
    i2=DATA_S2[number][1]
    i3=DATA_S2[number][2]
                   
    t=np.linspace(0,(len(v1)-1)*(1/fs),len(v1))  # vecteur temps\n",

    fig=plt.figure(figsize=(8,5),dpi=100)
    plt.plot(t[0:5000],v1[0:5000],lw=2,label='v1')
    plt.plot(t[0:5000],v2[0:5000],lw=2,label='v2')
    plt.plot(t[0:5000],v3[0:5000],lw=2,label='v3')
    plt.xlabel('t (s)')
    plt.ylabel('Voltage (V)')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.legend()
    plt.minorticks_on()      
    
    
    
    
    
    
    #########################################################################
    
    
    DATA_S=get_RTE_signal()
    number=10
    v1=DATA_S[number][0]
    v2=DATA_S[number][1]
    v3=DATA_S[number][2]
    i1=DATA_S[number][0]
    i2=DATA_S[number][1]
    i3=DATA_S[number][2]
                   
     
    t=np.linspace(0,(len(v1)-1)*(1/fs),len(v1))  # vecteur temps\n",
    
    fig=plt.figure(figsize=(10,5),dpi=100)
    plt.plot(t,v1/1000,lw=2)
    plt.plot(t,v2/1000,lw=2)
    plt.plot(t,v3/1000,lw=2)
    plt.xlabel('t [s]')
    plt.ylabel('Voltage (kV)')
    plt.title('voltages RTE')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    
    
    fig=plt.figure(figsize=(10,5),dpi=100)
    plt.plot(t,i1/1000,lw=2)
    plt.plot(t,i2/1000,lw=2)
    plt.plot(t,i3/1000,lw=2)
    plt.xlabel('t [s]')
    plt.ylabel('Current (kA)')
    plt.title('Currents RTE')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()



    #########################################################################


    v1,v2,v3,i1,i2,i3=get_EPRI_signal(21834)
    fs=15384.6 # samples frequency
     
    t=np.linspace(0,(len(v1)-1)*(1/fs),len(v1))  # vecteur temps\n",
    
    fig=plt.figure(figsize=(10,5),dpi=100)
    plt.plot(t,v1/1000,lw=2)
    plt.plot(t,v2/1000,lw=2)
    plt.plot(t,v3/1000,lw=2)
    plt.xlabel('t [s]')
    plt.ylabel('Voltage (kV)')
    plt.title('voltages RTE')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    
    
    fig=plt.figure(figsize=(10,5),dpi=100)
    plt.plot(t,i1/1000,lw=2)
    plt.plot(t,i2/1000,lw=2)
    plt.plot(t,i3/1000,lw=2)
    plt.xlabel('t [s]')
    plt.ylabel('Current (kA)')
    plt.title('Currents RTE')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()