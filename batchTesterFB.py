#Final implementation for FB multi-label tagging challenge
#The training set consists of nsamples~6e6 text questions and body
#and has about nlabels ~ 4e4 unique labels.  
#Our strategy will be to use a One-vs-Rest multi-label classifier
#that employs a Stochastic Gradient Descent binary classifier.
#However there are way too many labels, and we will need to choose
#some subset of labels for training -- follow the column subset selection problem:
#W. Bi and J. Kwok. "Efficient Multi-label Classification with Many Labels."
#jlmr.org 2013 vol.28
#Also we'll need to use training and predicting in batches

#-------------------------------------------------------------
#1)Extract tags : tagExtractFromCsv.py ---> taglist.pkl
#2)Generate subset of labels using CSSP: sparseLabelMatrix.py 
#----> Turns text labels to indicator label matrix with labelizer()
#----> {samp100, samp1000.pkl} used for subsetting: Ysub=Ytrain[:,samp100]
#where Yfull is ntrain x nlabels matrix (ntrain is batch sample size for training)
#----> {proj100, proj1000.pkl} are projector matrices to recover all labels
#proj100 is nlabels x 100 indicator matrix 
#The CSSP samples/projection matrix were generated using
#the first 1e6 samples of 6e6 Train data
#3)Define OvR multilabel classifier for batch SGD binary classifier: multilabelSGD.py
#4)Run the train/test in batches: batchTesterReutCorpus.py
#-------------------------------------------------------------
import numpy as np
import scipy
from sklearn.base import clone
from sklearn.metrics import hamming_loss, f1_score #accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
import nltk
import re
import itertools
import string
import pickle
import csv
import time


#Define parser/tokenizer for feature extractor
porter=nltk.PorterStemmer()#define my parser
punc=string.punctuation + '\n'+ '0123456789' #all punc+whtspc+newline
htmp=HashingVectorizer(stop_words='english')
sw=htmp.get_stop_words() #so we prebuild a sw list for speed in parser
regex = re.compile('[%s]' % re.escape(punc))

def myparserTwo(s):
	return [porter.stem(a) for a in regex.sub('',s).lower().split(' ') 
			if a not in sw and len(a)>2]


#Define a label indicator matrix creator (binarizer)
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

#Define OvR multilabel helper for SGD
def fitBinary(est,X,y,firstFit=0):
	if firstFit:
		estimator = clone(est)
		estimator.partial_fit(X,y,classes=[0,1])
	else:
		estimator=est
		estimator.partial_fit(X,y)
	return estimator

#Define a batch getter from csv file reader object 
def getBatch(size,doc_iter,testMode=False,start=0):
	data=[(doc[2],doc[3].split(' ')) for doc in 
			itertools.islice(doc_iter,start,start+size)]
	X, y = zip(*data) 
	if testMode: #return un-compressed labels
		return hasher.transform(X), y
	else:
		return hasher.transform(X), labelizer(y,all_classes)
		#return hasher.transform(X), lb.transform(y,all_classes)

#Define a generator using the above batch getter
def batchGen(size,doc_iter=None):
	X, y = getBatch(size,doc_iter) #need to do this to def stop condition
	while X.shape[0]: #stop condition here
		yield X, y #yield keyword defines a generator
		X, y = getBatch(size,doc_iter)

#Define a Testset batch getter from csv file reader object 
def getBatchTestSet(size,doc_iter,testMode=False,start=0):
	data=[(doc[0],doc[2]) for doc in 
			itertools.islice(doc_iter,start,start+size)]
	iden,X=zip(*data)
	return hasher.transform(X), iden

#Define a Test generator using the above batch getter
def batchGenTestSet(size,doc_iter=None):
	X, iden=getBatchTestSet(size,doc_iter) #need to do this to def stop condition
	while X.shape[0]: #stop condition here
		yield X,iden #yield keyword defines a generator
		X, iden=getBatchTestSet(size,doc_iter)



#Declare feature extractor
nfeat=2**18
hasher=HashingVectorizer(tokenizer=myparserTwo,
		decode_error='ignore',n_features=nfeat)
#Declare (single column of label matrix) multi-class classifier
clf=SGDClassifier()
#Load column indices of label matrix that we'll use for training
#samp1000=pickle.load(open('samp1000.pkl','rb'))
samp=pickle.load(open('samp1500.pkl','rb'))
#Load corresponding label projection matrix
proj=pickle.load(open('proj1500.pkl','rb')) #add switch for 100/1000

#Load a validation set from the last 1001 entries of training data
X_test=pickle.load(open('X_test.pkl','rb'))
y_test=pickle.load(open('y_test.pkl','rb'))
all_classes=pickle.load(open('all_classes','rb'))
#Binarize the labels to create a label matrix
#To use LabelBinarizer(), we need the entire
#training set ensure all classes are known
y_testNum=labelizer(y_test,all_classes)


#Need to densify and int for prediction comparisons
y_testBin=np.asarray(y_testNum.todense(),dtype='int')

#Create a csv-file iterator for training
csvfile=csv.reader(open('Train.csv','rU'),dialect=csv.excel)
csvfile.next() #pop-off the header

#Define the train batch size and declare iterator
#batchSizeTrain=10000
batchSizeTrain=1e4
batch_iterator=batchGen(size=batchSizeTrain,doc_iter=csvfile)

#Keep track of stats

stats={'n_train':0, 'accuracyHL': 0.0, 'F1':0.0, 'accuracy_history': [(0, 0,0)]
		, 't0': time.time(),'runtime_history': [(0, 0)]}

#To proceed without looping (for dev purposes)
#ebi=enumerate(batch_iterator)
#i, (X_train,y_train)=ebi.next()

#Main loop for batch training
for i, (X_train, y_train) in enumerate(batch_iterator):
	#Subset, densify, int label-mat
	y_trainSub=np.asarray(y_train[:,samp].todense(),dtype='int')
	#Define a list of estimators for each label (col of label-mat)
	if i == 0:
		firstCall=1 #First call to partial_fit
		ests=[fitBinary(clf,X_train,y_trainSub[:,j],firstFit=firstCall) 
				for j in range(y_trainSub.shape[1])]
	elif i > 1e2: # five loops for testing only
		print i
		print 'exiting training loop'
		print X_train.shape
		break
	else:
		firstCall=0
		ests=[fitBinary(ests[j],X_train,y_trainSub[:,j],firstFit=firstCall)
				for j in range(y_trainSub.shape[1])]

	#Make a prediction for each label
	preds=np.array([np.ravel(e.predict(X_test)) for e in ests])
	#Project the prediction to the full label space
	y_pred=proj.dot(preds)
	#Round to either 0 or 1 for each entry (decision)
	y_predRnd=(y_pred.transpose()>0.5).astype(int)
	#To get actual class labels in a list-of-lists (list for each row)
	#lolPred=[all_classes[np.ravel(np.nonzero(y_predRnd[k,:]))].tolist() 
		#	for k in range(y_predRnd.shape[0))]
	
	#Score our accuracy
	stats['n_train'] += X_train.shape[0]
	#stats['accuracy']=accuracy_score(y_testBin,y_predRnd)
	stats['accuracyHL']=hamming_loss(y_testBin,y_predRnd)*42048
	stats['F1']=f1_score(y_testBin,y_predRnd)
	stats['accuracy_history'].append((stats['accuracyHL'],
		stats['F1'], stats['n_train']))
	stats['runtime_history'].append((stats['accuracyHL'],time.time() - stats['t0']))
	print stats['n_train'], stats['accuracyHL'], stats['F1']
	if X_train.shape[0]<batchSizeTrain:
		break



#Now run to predict labels on the test set
testSet=csv.reader(open('Test.csv','rU'),dialect=csv.excel)
testSet.pop(0) # pop-off header
batchSizeTest=1e4
batch_iteratorTest=batchGenTestSet(size=batchSizeTest,doc_iter=testSet)
lolPred=[] #List-of-Lists predicted multi-label output
idenList=[]
for i, (X_testSet,iden) in enumerate(batch_iteratorTest):
	preds=np.array([np.ravel(e.predict(X_testSet)) for e in ests])
	#Project the prediction to the full label space
	y_pred=proj.dot(preds)
	#Round to either 0 or 1 for each entry (decision)
	y_predRnd=(y_pred.transpose()>0.5).astype(int)
	#To get actual class labels in a list-of-lists (list for each row)
	lolPred += [all_classes[np.ravel(np.nonzero(y_predRnd[k,:]))].tolist() 
			for k in range(y_predRnd.shape[0])]
	idenList += iden
	if X_testSet.shape[0]<batchSizeTest: #ends after last mini-batch of abnormal size
		print X_testSet.shape
		print 'exiting'
		break


#Write the predicted labels to a file
fo=open('predFB.csv','wb')
wr=csv.writer(fo)
wr.writerow(['Id','Tags']) #header
for i in range(len(lolPred)):
	wr.writerow([idenList[i],str(' '.join(lolPred[i]))])

fo.close()











