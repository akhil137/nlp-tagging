import sklearn
from sklearn.utils import multiclass
from sklearn import preprocessing
import numpy as np
import scipy
from scipy import sparse
from sparsesvd import sparsesvd
import pickle

all_classes=pickle.load(open('all_classes','rb'))
taglist=pickle.load(open('taglist.pkl','rb'))
taglist.pop(0)#pop off 'Tags' header

def labelizer(y, classes):
	imap = dict((v, k) for k, v in enumerate(classes))
	row=[]
	col=[]

        for i, lt in enumerate(y):
		for l in lt:
			if imap.get(l) is not None:
				row.append(i)
				col.append(imap[l])
	ra=np.array(row)
	ca=np.array(col)
	dat=np.ones_like(ra)
	#note dtype='float' below is good for svd and pinv, but not for classification
	Y=scipy.sparse.csc_matrix((dat,(ra,ca)),shape=(len(y),len(classes)),dtype='float')
        return Y

Yfull=labelizer(taglist,all_classes)

nsamples=1
Ytrunc=Yfull[1e5:2e5,:]

ksel=1500
u, s, vt=sparsesvd(Ytrunc,k=ksel)
#Now create projection matrix
#First create probability vector and sample k-of-d columns
def sampCols(ksel,p):
	np.random.seed(1979)
	samp=np.random.choice(p.size,size=1,p=p)
	while samp.size<ksel:
		tmp=np.random.choice(p.size,size=1,p=p)
		if not np.in1d(tmp,samp):
			samp=np.append(samp,tmp)
	return samp

p=1.0/(vt.shape[0])*vt.transpose().dot(vt).diagonal()
samp=sampCols(ksel,p) #k=ksel for sampling algo
Ysub=Ytrunc[:,samp]

#Create moore-penrose psuedo inverse
sinv=np.reciprocal(s)
YsubPinv=np.dot(vt.transpose(),np.dot(np.diag(sinv),u))

#Matrix-mult to create proj matrix 
#Gives seg fault
#Yt=Ytrunc.transpose()
#Ys=YsubPinv.transpose()
#proj=Yt.dot(Ys) 
#instead to dense multiply
proj=YsubPinv.dot(Ysub.todense()) #note this is written as np.matrix
fn='proj'+str(ksel)+'Only1e5Samples.pkl'
pickle.dump(proj,open(fn,'wb'))

#Do the following to load
#proj=pickle.load(open('proj2048Only1e5Samples.pkl','rb'))
#proj=np.asarray(proj)

#test projection
y_testBin=pickle.load(open('y_testBin_InScope.pkl','rb'))
Ysub=y_testBin[:,samp]
def err(k):
    tmp=Ysub[k,:].dot(proj.T)
    pred=(tmp>0.5).astype(int)
    #return absolute num incorrect labels per sample (hamming loss is normalized by #cols)
    return metrics.hamming_loss(y_testBin[k,:],pred)*42048

errvec=[err(k) for k in range(1001)]
errvecND=np.asarray(errvec)
import pylab as pl
bins=np.array([0,1,2,3,4,5,6]) 
#Also note that errvecND.max()=5; at most five errors
n, bins, patches=pl.hist(errvecND,bins)
