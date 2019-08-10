#Supporting functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import math

#load unsupervised dataset
def loadunsuperviseddata():
    OriginalXpd = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX = OriginalXpd.as_matrix()
    OriginalX = OriginalX.astype(float)
    return (OriginalX, OriginalXpd)

#load unsupervised dataset with test dataset
def loadunsuperviseddatawithtestdata():
    Xpd = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX = Xpd.as_matrix()
    OriginalX = OriginalX.astype(float)
    X_predictionpd = pd.read_csv("data_prediction.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX_prediction = X_predictionpd.as_matrix()
    OriginalX_prediction = OriginalX_prediction.astype(float)
    return (OriginalX, OriginalX_prediction, Xpd, X_predictionpd)
    
#load supervised dataset for classification
def loadsuperviseddataforclassification():
    datapd = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
    data = np.array(datapd)
    Originaly = np.ravel( np.c_[data[:,0]] )
    OriginalX = data[:,1:data.shape[1]].astype(float)
    data_prediction1pd = pd.read_csv("data_prediction1.csv", encoding='SHIFT-JIS', index_col=0)
    data_prediction1 = np.array(data_prediction1pd)
    Originaly_prediction1 = np.ravel( np.c_[data_prediction1[:,0]] )
    OriginalX_prediction1 = data_prediction1[:,1:data.shape[1]].astype(float)
    data_prediction2pd = pd.read_csv("data_prediction2.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX_prediction2 = np.array(data_prediction2pd)
    OriginalX_prediction2 = OriginalX_prediction2.astype(float)
    return (Originaly, OriginalX, Originaly_prediction1, OriginalX_prediction1, \
            OriginalX_prediction2, data_prediction2pd)

#load supervised dataset for regression
def loadsuperviseddataforregression():
    datapd = pd.read_csv("data.csv", encoding='SHIFT-JIS', index_col=0)
    data = np.array(datapd)
    data = data.astype(float)
    Originaly = np.c_[data[:,0]]
    OriginalX = data[:,1:data.shape[1]]
    data_prediction1pd = pd.read_csv("data_prediction1.csv", encoding='SHIFT-JIS', index_col=0)
    data_prediction1 = np.array(data_prediction1pd)
    data_prediction1 = data_prediction1.astype(float)
    Originaly_prediction1 = np.c_[data_prediction1[:,0]]
    OriginalX_prediction1 = data_prediction1[:,1:data.shape[1]]
    data_prediction2pd = pd.read_csv("data_prediction2.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX_prediction2 = np.array(data_prediction2pd)
    OriginalX_prediction2 = OriginalX_prediction2.astype(float)
    return (Originaly, OriginalX, Originaly_prediction1, OriginalX_prediction1, \
            OriginalX_prediction2, data_prediction2pd)
            
#Find variables with zero variance
def variableszerovariance( X ):
    Var0Variable = np.where( X.var(axis=0) == 0 )
    if len(Var0Variable[0]) == 0:
        print( "No variables with zero variance" )
    else:
        print( "{0} variable(s) with zero variance".format(len(Var0Variable[0])))
        print( "Variable number: {0}".format(Var0Variable[0]+1) )
        print( "The variable(s) is(are) deleted." )
    return Var0Variable

#Save matrix
def savematrixcsv( X, index, filename):
    Xpd = pd.DataFrame(X)
    Xpd.index = index
    exec("Xpd.to_csv( \"{}.csv\", header = False )".format( filename ) )

def savematrixcsv2( X, index, column, filename):
    Xpd = pd.DataFrame(X[:, np.newaxis])
    Xpd.index = index
    Xpd.columns = column
    exec("Xpd.to_csv( \"{}.csv\" )".format( filename ) )

#Save matrix with column name
def savematrixcsvwithcolumnname( X, index, columnname, filename):
    Xpd = pd.DataFrame(X)
    Xpd.index = index
    Xpd.columns = columnname
    exec("Xpd.to_csv( \"{}.csv\" )".format( filename ) )
    
#make scatter plot
def scatterplotwithsamplename(x, y, xname, yname, samplename, clusternum=0):
    if clusternum==0:
        plt.scatter(x, y)
    else:
        plt.scatter(x, y, c=clusternum, cmap=plt.get_cmap('jet'))
        
    for numofsample in np.arange( 0, samplename.shape[0]-1):
        plt.text(x[numofsample], y[numofsample], samplename[numofsample], horizontalalignment='left', verticalalignment='top')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
    
#make tt plots for PCA with clustering result
def makettplotwithclustering(ScoreT, ClusterNum, Xpd):
    plt.scatter(ScoreT[:,0], ScoreT[:,1], c=ClusterNum, cmap=plt.get_cmap('jet'))
    for numofsample in np.arange( 0, ScoreT.shape[0]-1):
        plt.text(ScoreT[numofsample,0], ScoreT[numofsample,1], Xpd.index[numofsample], horizontalalignment='left', verticalalignment='top')
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()

#Calculate r^2
def calc_r2( ActualY, EstimatedY ):
    return float( 1 - sum( (ActualY-EstimatedY )**2 ) / sum((ActualY-ActualY.mean(axis=0))**2) )

#Calculate RMSE
def calc_rmse( ActualY, EstimatedY ):
    return( math.sqrt( sum( (ActualY-EstimatedY )**2 ) / ActualY.shape[0]) )

#Make YYplot
def make_yyplot( ActualY, EstimatedY, YMax, YMin, EstimatedYName ):
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(ActualY,EstimatedY)
    plt.plot([YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], [YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], 'k-')
    plt.ylim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
    plt.xlim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
    plt.xlabel("Actual Y")
    plt.ylabel(EstimatedYName)
    plt.show()

#Make Threshold for T^2 and SPE
def make_threshold_t2spe( Index, NumOfIndexThreshold ):
    SortedIndex = np.sort(Index)
    return( SortedIndex[NumOfIndexThreshold-1] )

#make time plot with threshold
def make_timeplot_with_threshold(Index, Threshold, xname, yname):
    plt.plot(Index, 'ko')
    plt.plot([0,Index.shape[0]], [Threshold,Threshold], 'r-')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

#Optimize gamma to maximize variance of Gaussian gram matrix
def optimize_gamma_grammatrix(X, CandidatesOfGamma):
    # Calculate gram matrix of Gaussian kernel and its variance for each gamma candidate
    VarianceOfKernelMatrix = list()
    for CandidateOfGamma in CandidatesOfGamma:
        KernelMatrix = np.exp(-CandidateOfGamma*((X[:, np.newaxis] - X)**2).sum(axis=2))
        VarianceOfKernelMatrix.append(KernelMatrix.var(ddof=1))
    # Decide the optimal gamma with the maximum variance value
    return CandidatesOfGamma[ np.where( VarianceOfKernelMatrix == np.max(VarianceOfKernelMatrix) )[0][0] ]
