#Use the code from sklearn.preprocessing.label.py (label.binarize) 
#but output a scipy.sparse matrix (format can be trivially changed)
import csv
import sklearn
from sklearn.utils import multiclass
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import scipy
from scipy import sparse

#Initially read some labels from Train.csv
nsamples=5 #define num-rows to read

csvfile=csv.reader(open('subTrain.csv','rU'),dialect=csv.excel)
csvfile.next() #pop off header

taglist=[] #holds list of lists for multi-label tags
#for i in range(nsamples):
#	taglist.append(csvfile.next()[3].split(' '))
for row in csvfile:
	taglist.append(row[3].split(' '))

#or do taglist=pickle.load(open('taglist.pkl','rb')) #6.03x10^6
taglist.pop(0) #needed to pop off header
all_classes=multiclass.unique_labels(taglist) #42,048 in full set

#Here's wher we start cribbing from label.binarize()
imap=dict((v,k) for k, v in enumerate(all_classes))

row=[]
col=[]
#now start to create row/col indices for sparse mat
for i, lt in enumerate(taglist):
	for l in lt:
		row.append(i)
		col.append(imap[l])


ra=np.array(row)
ca=np.array(col)
dat=np.ones_like(ra)

#***Just use labelizer function instead
#Now fill out sparse matrix in csr format
#Y=sparse.csr_matrix((dat,(ra,ca)),shape=(nsamples,len(classes)))
#Yfull=sparse.csc_matrix((dat,(ra,ca)),shape=(nsamples,len(classes)))
#Yfull=sparse.csc_matrix((dat,(ra,ca)),shape=(len(taglist),len(all_classes)))

#--To test if this we worked 
#--turn above sparse to dense
#Yd=Y.todense()
#--Then check with the std method in sklearn
#lb=preprocessing.LabelBinarizer()
#Ylb=lb.fit_transform(taglist)
#--Check if the two dense arrays are equal
#np.array_equal(Yd,Ylb)

#Get k highest occurence columns
#kcols=1000
#rsum=Yfull.sum(0)
#rsumargsort=rsum.argsort()
#colarglist=sum(rsumargsort.tolist(),[]) #flatten to list
#Ysub=Yfull[:,colarglist[-kcols:]]

#Now that we have the column indices for k highest occurences
#map them back to the actual labels (reverse dict lookup)
#subclasses=[]
#for k, v in imap.iteritems():
#	if v in colarglist[-kcols:]:
#		subclasses.append(k)


#We can now create a sparse label matrix (densify later if needed) per batch
#using Yhat=labelizer(y,subclasses)

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



#Note a svd of Yfull needs to know rank ahead of time (e.g. need k ahead of time)
Yfull=labelizer(taglist,all_classes) 
nsamples=100000 #gone up to 1e6 samples
Ytrunc=Yfull[:nsamples,:] #perform svd on truncated Yfull to save time
krank=10
#---WARNING: SLOW AS A MOFO----
#for Ytrunc 1e6x4.2e5
#%time u, s, vt=sparse.linalg.svds(Ytrunc,k=10)
#CPU times: user 41.71 s, sys: 1.23 s, total: 42.94 s
#Wall time: 4.81 s
#%time u, s, vt=sparse.linalg.svds(Ytrunc,k=100)
#CPU times: user 235.91 s, sys: 5.40 s, total: 241.31 s
#Wall time: 26.05 s

u, s, vt=sparse.linalg.svds(Yfull,k=krank) #LHS not sparse objects

#much faster to use sparsesvd
#https://pypi.python.org/pypi/sparsesvd
#installed on vincent: pip ./ (inside the dir) - with Canopy's pip

from sparsesvd import sparsesvd
u, s, vt=sparsesvd(Ysub,k=10)

#note vt.shape=(krank,len(all_classes))
#these were pickle.dumped on vincent vt100 and vt1000
#we should also dump u, s so that we can compute pinv more quickly!
sig=scipy.linalg.diagsvd(s,krank,krank)
#Not needed: Ysub=u.dot(sig.dot(vt)) #matrix multiplication


#Column selection using probabilities p_i=1/k||v_t_(i)||^2
#note p.size=total number of labels=len(all_classes)
p=1.0/(vt.shape[0])*vt.transpose().dot(vt).diagonal()
ksel=krank #has to be according to Bi/Kwok vol. 28 jlmr

def sampCols(ksel,p):
	np.random.seed(1979)
	samp=np.random.choice(p.size,size=1,p=p)
	while samp.size<ksel:
		tmp=np.random.choice(p.size,size=1,p=p)
		if not np.in1d(tmp,samp):
			samp=np.append(samp,tmp)
	return samp
#now sample columns - note sampCols sets seed!
samp100=sampCols(100,p100)
samp1000=sampCols(1000,p1000)
#pickle.dump(samp1000,...)
#Now we're going to apply the subsetting to mini-batches!
#have fun looking at popular tags
#all_classes[sampCols(10,p).tolist()]
Ysub=Yfull[:,samp100]
#Moore Penrose psuedo inv
YsubPinv=np.linalg.pinv(np.asarray(Ysub.todense()))

#To create the projection matrix e.g. k-by-d matrix
#much faster to use spars.dot(np) rather than np.dot(sparse.todense())
Yt=Ytrunc.transpose()
Ys=YsubPinv.transpose()
proj100=Yt.dot(Ys) #note proj.shape=(d,k=100) - d>>k
#do the same for proj1000, using Ysub=Yfull[:,samp1000] and pinv

#fit-to-sub, predict, re-project to full, rnd,score
clf=OneVsRestClassifier(SGDClassifier())
y_testSub=y_testBinNDArray[:,samp]
clf.fit(X_test,y_testSub) #note if we fit to y_testBinNDArray
h_pred=clf.predict(X_test) #then metrics below to h_pred is perfect 1.0
#y_pred=h_pred.dot(YsubPinv.dot(np.asarray(Yfull.todense())))
y_pred=h_pred.dot(proj100.transpose())
y_predRnd=(y_pred>0.5).astype(int)
metrics.accuracy_score(y_testBinNDArray,y_predRnd)


#Notes: how large of a k to keep
#numerical experiment
nsamples,nlabels=1000,100
y=np.random.randint(0,2,size=[nsamples,nlabels])
ys=scipy.sparse.csc_matrix(y)
ksel=50
ut,s,vt=sparsesvd(ys,k=ksel)
p=1.0/(vt.shape[0])*vt.transpose().dot(vt).diagonal()
samp=sampCols(ksel,p)
ytmp=y.copy()
ytmp[:,samp]=0 # mask out selected columns
ydiff=y-ytmp #Now all that remains are selected columns
metrics.accuracy_score(y,ydiff) #If all the predictions

#Best expected error performance: forget classification, just test projection/encoding

Ytrunc=Yfull[:1e6,:].astype(int)
Ysub=Ytrunc[:,samp1000]
proj1000T=proj1000.transpose()
def err(k):
    #tmp=Ysub[k,:].dot(proj1000T)
    tmp=Ysub[k,:].dot(proj)
    pred=(tmp>0.5).astype(int)
    #return absolute num incorrect labels per sample (hamming loss is normalized by #cols)
    #return metrics.hamming_loss(Ytrunc[k,:].todense(),pred)*42048
    return metrics.hamming_loss(y_testBin[k,:],pred)*42048
#Takes a while
errvec=[err(k) for k in range(100)]
errvecND=np.asarray(errvec)

#To compare this to randomly selecting a 1000 columns from Ytrunc
#we'd need to re-svd to generate a projection matrix
#rnd1000=np.random.choice(Ytrunc.shape[1],size=1000,replace=False)


#WE can count how many times we have captured (at-least one) a samp1000 (CSSP) label
YsubFull=Yfull[:,samp1000]
np.count_nonzero(YsubFull.sum(1)) #5561708; out of 6034195 samples, e.g. 92%

#For each sample we have 0-5 errors
#Using the first 1e5 samples, let's create an errvec and look at hamming loss
import pylab as pl
bins=np.array([0,1,2,3,4,5,6]) #Note Yfull.sum(1).max()=5, e.g. at most 5 labels
#Also note that errvecND.max()=5; at most five errors
n, bins, patches=pl.hist(errvecND,bins)


#For each subset of samples where we have x errors, find the label indices
errSample=(errvecND==5).astype(int) #boolean where there are x=5 errors 
errSampleIndx=np.flatnonzero(errSample) #sample index where there are x=5 errors
#Count the number of times the samp1000 labels appear (e.g. when Ysub[k,:] has a 1)
countLabels=[np.nonzero(Ysub[k,:])[1].shape[0] for k in errSampleIndx]
#For example, with x=5 errors, we get that countLabels=0 for all such errors
#np.asarray(countLabels).max()=0 -- i.e. none of the samp1000 columns are present
#With x=1, most of the samples have a nonzero number of samp1000 columns
pl.hist(np.asarray(countLabels),bins) #3483 times zero of the samp1000
				      #10389 times one ...
				      #11809 times two
				      #6948 times three
				      #3462 times four
				      #4 times five

#Consider the one errors, and examine how many times the incorrectly predicted
#label is not part of samp1000, i.e. count the non-overlap with samp1000
def count1ErrNonOverlap(k):
    return np.abs(np.in1d(np.nonzero(Ytrunc[k,:])[1],samp1500)-1).sum()


counts1ErrNonOvrLap=[count1ErrNonOverlap(k) for k in errSampleIndx]
c1errNO=np.asarray(counts1ErrNonOvrLap)#min=0, max=2; 
#Even when we have 0 non-overlap, e.g. all labels are part of samp1000, we still
#incur 1 error, and sometimes we have 2 labels not part of samp1000, but only
#incur 1 error (good, we predicted a non-overlapping label)
c1errNOargsort=c1errNO.argsort()
#c1errNOargsort[-1] = 17120 #index of errSampleIndx vector that has 2 non-overlaps
#but only 1 error
np.in1d(np.nonzero(Ytrunc[errSampleIndx[17120],:])[1],samp1000) #F, F, T
#errSampleIndx[17120]=47351
yhat=Ysub[47351,:].dot(proj1000T)
yhatargsort=yhat.argsort()
#The labels that should be predicted are np.nonzero(Ytrunc[errSampleIndx[17120],:])[1]
#[13279, 29101, 37004], with first two having non-overlap with samp1000 (F,F,T)
#The value of yhat's closest to 1 are
#yhat[0,yhatargsort[0,-1]], and yhat[0,yhatargsort[0,-2]]
#corresponding to labels [37004, 29101] - so we predicted 29101 even though it is not
#part of samp1000, however we could not predict 13279

#we can perform the same excercise when we have zero non-overlap and 1-error
#e.g. c1errNO[c1errNOargsort[0]=3392]=0 -> zero overlap
#Should predict np.nonzero(Ytrunc[errSampleIndx[3392],:])[1] = [20561,37004]
#both of which are in samp1000 (hence zero non-overlap), 
#however we predict an extra label: yhat[0,yhatargsort[0,-3]]=0.83
#yhatargsort[0,-3]=29101 which is not part of samp1000



#distribution of number of non-overlaps for 1-errors
bins=np.array([0,1,2,3])
n, bins, patches=pl.hist(c1errNO,bins)

def pd(k):
	tmp=YsubCSR[k,:].dot(proj1000T)
	pred=(tmp>0.5).astype(int)
	return pred



