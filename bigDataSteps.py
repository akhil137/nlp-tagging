#imagine you have a huge data file (say csv for now)

import csv
#open the file
r=csv.reader(open('Train.csv','rU'),dialect=csv.excel)
#note that r implements the iterator protocol, e.g. ther is a r.next()

#to read only xnum of lines use itertools(?)
from itertools import islice
id, title, body, label = [],[],[],[]
for row in islice(r,batch_size):
	#do stuff
	print row


#-----------------------
#out of core learning
#-----------------------
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np

batch_size=100 #num lines
nfeat=2**18 #num of features for hasher
clf=SGDClassifier() #define classifier
#define feature extracter
hasher=HashingVectorizer(tokenizer=myparser,decode_error='ignore',n_features=nfeat)



#--------------------------------------------------------
#Ideally we'd like to create a generator of corpus tuples:
#corpusGen=((reuters.raw(fileid),cat) for cat in reuters.categories() 
#		for fileid in reuters.fileids(cat))
#BUT how do we randomly split the corpus tuples into train/test sets?
#So we take the brute-force approach and create an offline 
#randomized corpus with test set precedding train set
#using code from ipyTxtClassSetEnv.py
corpus=[(reuters.raw(fileid),cat) for cat in reuters.categories()
for fileid in reuters.fileids(cat)]
random.seed(1979)
random.shuffle(corpus)
testSize=int(len(corpus)*0.1)
train_raw=corpus[testSize:]
test_raw=corpus[:testSize]

#concatenate test+train and then strip off only testSize 
splitCorpus=test_raw+train_raw
#-----------------------------------------------------------
#make a generator for the corpus
corpusGen=((doc,tag) for (doc,tag) in splitCorpus)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(reuters.categories())

#For multi-label classification, fake it for now
#fake for now - one class distinction
pos_class='python'


#batch iterator
#argument doc_iter can be a csv.reader object
#argument size is batch size
def getBatch(size,doc_iter=None):
	#default to reuters corpus
	if doc_iter is None:
		data=[(a,b) for (a,b) in itertools.islice(corpusGen,size)]	
		X, y = zip(*data) #Note '*' operator unpacks a list
		return hasher.transform(X), le.transform(y)
	else: #otherwise pass a csv-reader object
		data=[(doc[2],pos_class in doc[3]) for doc in 
				itertools.islice(doc_iter,size)]
		X, y = zip(*data) #Note '*' operator unpacks a list
		return hasher.transform(X), np.asarray(y,dtype=int)

#Now define a generator for batches
def batchGen(size,doc_iter=None):
	X, y = getBatch(size,doc_iter)
	while X.shape[0]:
		yield X, y #yield keyword defines a generator
		X, y = getBatch(size,doc_iter)



#for csv file
all_classes=np.array([0,1])
X_test, y_test = getBatch(size=testSize,doc_iter=r)
batchSizeTrain=10
batch_iterator=batchGen(size=batchSizeTrain,r)

#for reuters corpus
all_classes=le.transform(le.classes_)
X_test, y_test = getBatch(size=testSize)
batchSizeTrain=10
batch_iterator=batchGen(size=batchSizeTrain)

#Define a dictionary of stats to keep track off
#to rerun if failed, must do the following three steps
#clf==SGDClassifier() 
#batch_iterator=batchGen(size=batchSizeTrain)
#stats={'n_train':0, 'accuracy': 0.0}

stats={'n_train':0, 'accuracy': 0.0}


#main loop: iterate on batches
for i, (X_train, y_train) in enumerate(batch_iterator):
	clf.partial_fit(X_train,y_train,classes=all_classes)
	stats['n_train'] += X_train.shape[0]
	stats['accuracy'] = clf.score(X_test,y_test)
	
	if i % 10 == 0:
		print stats


#prediction loop - this can be done line by line
#probaly just use the csv module to read the pred set
#and write to it as well
for i, X_test in enumerate (pred_iterator):


		

