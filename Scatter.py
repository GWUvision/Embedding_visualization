import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def recall(Fvec, label):
    """This function is used to find the nearest same-class points similarity for each instance 
    and neareat different-class similarity points for each instance.

    Parameters
    ----------

    Fvec : Tensor, shape (n_samples, n_features)
    Embedding representation.
    label : List, length (n_feafures)
    Label for each instance.

    Returns
    -------

    disList : List, shape (n_features,2)
    Two similarity(nearest same-class and nearest diff-class) for each instance. 
    """  
    N = len(label)
    print(N)
    disList = []
    
    D = Fvec.mm(torch.t(Fvec))

    D[torch.eye(N).byte()] = -1
    
    sort_D, indices = torch.sort(D,1,descending=True)
    print(indices[0])
    for i in range(0,N):
        dis = []
        distance = sort_D[i]
        index = indices[i]
        
        for j in range(0,N):
            if label[index[j].item()]==label[i]:
                dis.append(distance[j].item())
                break;
        
        for k in range(0,N):
            if label[index[k].item()]!=label[i]:
                dis.append(distance[k].item())
                break;
            
        disList.append(dis)
    return disList

def draw_sca(disList,name='',savepath='./'):
    """ This function is used to generate similarity scatter plot and save it.

    Parameters
    ----------

    disList : List, shape (n_features,2)
    Two similarity(nearest same-class and nearest diff-class) for each instance. 
    
    name : String
    Title of scatter plot
    
    savepath : String
    Path to save this plot
    
    """  
    
    disList = np.asarray(disList)
    print(disList.shape)
    A=disList[:,0]
    B=disList[:,1]
    diff=B>A
    same=A>B
    plt.figure(figsize=(6.4,6.4))
    plt.title(name)
    plt.xlim(0.3,1.0)
    plt.ylim(0.3,1.0)
    plt.xlabel('Similarity (Same class)',fontsize=22)
    plt.ylabel('Similarity (Different class)',fontsize=22)
    samex = A[same]
    samey = B[same]
    difx = A[diff]
    dify = B[diff]
    plt.scatter(samex,samey,c='c',s=0.1)
    plt.scatter(difx,dify,color=[1.0, 0.5, 0.25],s=0.1)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    plt.savefig(savepath+'scatter_plot.jpg')
    
def gene_sca(vecPath,labelPath,name='',savepath='./'):
    """ This function is used to find generate similarity list, generate similarity scatter plot and save it.

    Parameters
    ----------

    vecPath : String
    Path saving embedding vector
    
    labelPath : String
    Path saving label
    
    name : String
    Title of scatter plot
    
    savepath : String
    Path to save this plot

    """  
    #Fvec = torch.load(vecPath)
    Fvec = np.load(vecPath)
    #label = torch.load(labelPath)
    label = np.load(labelPath)
    Fvec = torch.Tensor(Fvec)
    disList = recall(Fvec,label)
    draw_sca(disList,name,savepath)