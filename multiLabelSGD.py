#Want to perform multi-label classification with SGD
#SGD natively just does multi-class (one ouptput per feature)
#The OneVsRestClassifier does multi-label but can't do partial-fit
#So here we hack it together

#The trick is to look at _fit_binary(est,X,y,classes) in multiclass.py in sklearn 0.14
##Here we assume that X is [nsamples,nfeatures]; and y is [nsamples,1]
#Only the else gets executed normally
#estimator = clone(estimator) -- where we have from sklearn.base import clone
#estimator.fit(X,y) -- so the trick is just to create a clone!!

#Then we follow the fit_ovr(estimator,X,Y,n_jobs) method - one.vs.rest
#the main idea is to use list comprehension to define a bunch of estimators
#one for each column of Y, which is [nsamples,nlabels]

#Lastly we follow predict_binary and _predict_ovr and use array flattening
#and list comprehension to make our Yhat
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import scipy
def fitBinary(est,X,y,firstFit=0):
	if firstFit:
		estimator = clone(est)
		estimator.partial_fit(X,y,classes=[0,1])
	else:
		estimator=est
		estimator.partial_fit(X,y)
	return estimator

#the assuming we're inside the batching loop
#create a list of estimators which are each cloned and fit for each column of Y
#first call to partial_fit
ests=[fitBinary(clf,X,Y[:,i],firstFit=1) for i in range(Y.shape[1])]

#latter call to partial fits
ests=[fitBinary(e,X,Y[:,i])
		for e in ests
		for i in range(Y.shape[1])]



#then create an np array of predictions using the above estimators
preds=np.array([np.ravel(e.predict(X)) for e in ests])

#for testing with random enteries 
np.random.seed(1979)
nsamples=100
nfeatures=1000
numOnesInX=np.int(0.01*nsamples*nfeatures) #1% sparsity
nlabels=4
ra=np.random.randint(0,nsamples,size=numOnesInX) #nsamples=10
ca=np.random.randint(0,nfeatures,size=numOnesInX) #nfeatures=1000
dat=np.ones_like(ra)
X=scipy.sparse.csc_matrix((dat,(ra,ca)),shape=(nsamples,nfeatures))
y=np.random.randint(0,2,size=[nsamples,nlabels]) #nlabels=4

#test for single-label
clf=SGDClassifier()
clf.fit(X,y[:,0])
pred=clf.predict(X)
np.array_equal(pred,y[:,0]) #check true or false
accuracy_score(pred,y[:,0]) #or get an accuracy score

#Note that fit and partial_fit don't always give the same result???
#reserve first 10 samples for test

clf=SGDClassifier()
clf.fit(X[11:99,:],y[11:99,0]) #vs. partial_fit (just re-run from clf=...)
pred=clf.predict(X[:10,:])
accuracy_score(pred,y[:10,0]) #0.4 for fit and 0.5 for partial_fit

#another oddity: even if we test and train samples overlap
#partial_fit must be called twice (with same data) to get 1.0 accuracy_score
clf=SGDClassifier()
clf.partial_fit(X[:99,:],y[:99,0],classes=[0,1])
pred=clf.predict(X[:10,:])
accuracy_score(pred,y[:10,0]) #gives 0.8
clf.partial_fit(X[:99,:],y[:99,0]) #fit again on same data!
pred=clf.predict(X[:10,:])
accuracy_score(pred,y[:10,0]) #now we get 1.0
