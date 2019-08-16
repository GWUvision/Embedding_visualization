import numpy as np
#import torch
from sklearn import manifold
import os
#import torch
import Yoked_Tsne
import matplotlib.pyplot as plt


def draw(comEmbed,traLabel,valLabel,name,path,i):
    """ This function is used to draw joint tsne plot.

    Parameters
    ----------

    comEmbed : Tensor, shape (n_samples, n_features)
    Embedding representation of both training data and testting data.
    
    traLabel : List, length (n_tra_feafures)
    Label for each training instance.
    
    valLabel : List, length (n_tra_feafures)
    Label for each testting instance.
    
    name : String,
    Title for this joint tsne plot.
    
    path : String,
    Path to save this plot."""

    trasize = len(traLabel) 
    a3=np.append(traLabel,valLabel)%7
    b3 = np.append(traLabel,valLabel)%12
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","x","1","2","3","4"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    colorslist3=[]
    markerslist3 = []
    for item in a3:
        colorslist3.append(colors[item])
    for item in b3:
        markerslist3.append(markers[item])
        
    plt.figure(figsize=(90,30))
    plt.axis((-250,250,-250,250))
    plt.title(name)
    
    axe1=plt.subplot(1,3,1)
    axe1.set_title('tra')
    
    axe2=plt.subplot(1,3,2)
    axe2.set_title('All')
            
    axe3=plt.subplot(1,3,3)
    axe3.set_title('val')
    
    for k in range(0,trasize):
        axe1.scatter(comEmbed[k, 0], comEmbed[k, 1],c=colorslist3[k],marker=markerslist3[k])
        axe1.axis('on')
        axe2.scatter(comEmbed[k, 0], comEmbed[k, 1],c='r',marker=markerslist3[k],alpha=0.5)
        axe2.axis('on')
    for k in range(trasize,len(comEmbed)):
        axe2.scatter(comEmbed[k, 0], comEmbed[k, 1],c='b',marker=markerslist3[k],alpha=0.5)
        axe2.axis('on')
        axe3.scatter(comEmbed[k, 0], comEmbed[k, 1],c=colorslist3[k],marker=markerslist3[k])
        axe3.axis('on')
        
    plt.savefig(path+'Joint-sne_'+str(i)+'.jpg',dpi=300)
    plt.close()
    

    
def jointTsne(traPathList,valPathList,traLabelPath,valLabelPath,name=[''],path='./',alpha=8):
    
    """ This function is used to draw joint tsne plot.

    Parameters
    ----------

    traPath : String
    Path saving training embedding vectors
    
    valPath : String
    Path saving testing embedding vectors
    
    traLabelPath : String
    Path saving training label
    
    valLabelPath : String
    Path saving testing label
    
    name : String
    Title for this joint tsne plot.
    
    path : String
    Path to save this plot.
    
    alpha : int
    Hyperparameter for mapping cluster in different plot in together. Default:8
    
    """
    
    #traFvec = torch.load(traPathList[0])
    traFvec = np.load(traPathList[0])
    #valFvec = torch.load(valPathList[0])
    valFvec = np.load(valPathList[0])
    #traLabel = np.asarray(torch.load(traLabelPath))
    traLabel = np.load(traLabelPath)
    #valLabel = np.asarray(torch.load(valLabelPath))
    valLabel = np.load(valLabelPath)
    
    comFvec = np.concatenate((traFvec, valFvec), axis=0)
    comEmbed_0 = manifold.TSNE(n_components=2).fit_transform(comFvec)
    draw(comEmbed_0,traLabel,valLabel,name[0],path,'')
    
    for i in range(1,len(traPathList)):
        #traFvec = torch.load(traPathList[i])
        traFvec = np.load(traPathList[i])
        #valFvec = torch.load(valPathList[i])
        valFvec = np.load(valPathList[i])
        
        comFvec = np.concatenate((traFvec, valFvec), axis=0)
        comEmbed,_=Yoked_Tsne.Yoked_TSNE(n_components=2,init_Y=comEmbed_0).Yoke_transform(comFvec,0,alpha=10**-alpha,fixed_Y=True)
        draw(comEmbed,traLabel,valLabel,name,path,i)