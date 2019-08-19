# Embedding_visualization
Code for ICML2019 workshop(Understanding and Improving Generalization in Deep Learning) paper: Visualizing How Embeddings Generalize
---


arXiv link:

Reqirement: python3, numpy, scikit-learn, matplotlib, scipy, pytorch

This code contains implementation of two method in paper:\
1.Joint-tsne:
#
```python
import Jointsne
Jointsne.jointTsne(traPathList,valPathList,traLabelPath,valLabelPath,name,path,alpha)
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
```
2.Scatter plot:
#
```python
import Scatter
Scatter.gene_sca(vecPath,labelPath,name,savepath):
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
```
