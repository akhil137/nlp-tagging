#-----------------------
#out of core learning
#-----------------------
import csv
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn import metrics
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords 

import numpy as np
import re
import itertools
import string

porter=nltk.PorterStemmer()#define my parser
punc=string.punctuation + '\n'+ '0123456789' #all punc+whtspc+newline
#swSet=set(stopwords.words('english')) #nltk stopword list smaller than sklearn
htmp=HashingVectorizer(stop_words='english')
sw=htmp.get_stop_words() #so we prebuild a sw list for speed in parser
#table=string.maketrans("","")

regex = re.compile('[%s]' % re.escape(punc))

#def myparser(s):
#	punc='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n ]' #all punc+whtspc+newline
#	np=[a for a in re.split(punc,s) if a not in string.punctuation]
#	low=[a.lower() for a in np if len(a)>2] #only two-lett words lowered
#	nostop=[a for a in low if a not in stopwords.words('english')]
#	return [porter.stem(a) for a in nostop if re.findall(r"[^\W\d]",a)]

#This parser is almost 10x faster than above and gives rougly same output
#e.g. number of features are the same
def myparserTwo(s):
	#translate() doesn't work the same for unicode and non-unicode
	#return [porter.stem(a) for a in s.translate(table,punc).split(' ') if a not in swSet and len(a)>2]
	#thus replace punctuation with regex.sub as such
	return [porter.stem(a) for a in regex.sub('',s).lower().split(' ') if a not in sw and len(a)>2]


batch_size=100 #num lines
nfeat=2**18 #num of features for hasher
clf=OneVsRestClassifier(SGDClassifier()) #define classifier
clf2=MultinomialNB()

#define feature extracter
#using pre-built sw extractor in parser is faster (slightly) then letting vectorizer do it
#hasher=HashingVectorizer(tokenizer=myparser,decode_error='ignore',n_features=nfeat, stop_words='english')
hasher=HashingVectorizer(tokenizer=myparserTwo,decode_error='ignore',n_features=nfeat)
#hasher=HashingVectorizer(tokenizer=myparserTwo,decode_error='ignore',n_features=nfeat, non_negative=True)



#--------------------------------------------------------
#Ideally we'd like to create a generator of corpus tuples:
#corpusGen=((reuters.raw(fileid),cat) for cat in reuters.categories() 
#		for fileid in reuters.fileids(cat))
#BUT how do we randomly split the corpus tuples into train/test sets?
#So we take the brute-force approach and create an offline 
#randomized corpus with test set precedding train set
#using code from ipyTxtClassSetEnv.py


#JUST UNPICKLE if you can splitCorpus=pickle.load(open('splitCorpus.pkl','rb'))

#reutDocs=[(reuters.raw(fileid) ,cat)
#for cat in reuters.categories()
#for fileid in reuters.fileids(cat)]

#corpus=[(a,b) for (a,b) in reutDocs]
#random.seed(1979)
#random.shuffle(corpus)
#testSize=int(len(corpus)*0.1)
#train_raw=corpus[testSize:]
#test_raw=corpus[:testSize]

#concatenate test+train and then strip off only testSize 
#splitCorpus=test_raw+train_raw
#-----------------------------------------------------------
#Choose either reuters corpus or csv file
#splitCorpus=pickle.load(open('splitCorpus.pkl','rb'))
csvfile=csv.reader(open('subTrain.csv','rU'),dialect=csv.excel)

#make a generator for the corpus
#corpusGen=((doc,tag) for (doc,tag) in splitCorpus)

#process the tags
#for reuterts single labels ok to use LabelEncoder()
#le=preprocessing.LabelEncoder()
#le.fit(reuters.categories())
#tagset=pickle.load(open('tagset.pkl','rb'))
#allclasses=list(tagset)

#for fb data, with multilabels, must use LabelBinarizer()
le=preprocessing.LabelBinarizer()
taglist=pickle.load(open('taglist.pkl','rb'))
taglist.pop(0) #pop-off header
le.fit(taglist)

#For multi-label classification, fake it for now
#fake for now - one class distinction
#pos_class='python'


#batch iterator
#argument doc_iter can be a csv.reader object
#argument size is batch size
def getBatch(size,doc_iter=None,testMode=False,start=0):
	#default to reuters corpus
	if doc_iter is None:
		data=[(a,b) for (a,b) in itertools.islice(corpusGen,size)]
		if len(data):
			X, y = zip(*data) #Note '*' operator unpacks a list
			return hasher.transform(X), le.transform(y)
		else: 
			return X
	else: #otherwise pass a csv-reader object
		data=[(doc[2],doc[3].split(' ')) for doc in 
				itertools.islice(doc_iter,start,start+size)]
		X, y = zip(*data) #Note '*' operator unpacks a list
		#return hasher.transform(X), le.transform(y)
		#using labelizer functionin sparseLabelMatrix.py
		if testMode: #return un-compressed labels
			return hasher.transform(X), y
		else:
			return hasher.transform(X), labelizer(y,all_classes)


#Now define a generator for batches
def batchGen(size,doc_iter=None):
	X, y = getBatch(size,doc_iter) #need to do this to def stop condition
	while X.shape[0]: #stop condition here
		yield X, y #yield keyword defines a generator
		X, y = getBatch(size,doc_iter)





#for reuters corpus
#testSize=int(len(splitCorpus)*0.1)

#all_classes=le.transform(le.classes_)
#X_test, y_test = getBatch(size=testSize)
#batchSizeTrain=1024
#batch_iterator=batchGen(size=batchSizeTrain)


#all_classes=le.transform(le.classes_)
#faster
#all_classes=range(len(le.classes_))

#We should let y_test be the full nsamples-by-d matrix
#get the last 9999 instead
#X_test, y_test = getBatch(size=testSize,doc_iter=csvfile,testMode=True,start=len(taglist)-10000)
#X_test, y_test = getBatch(size=testSize,doc_iter=csvfile,testMode=True)
csvfile=csv.reader(open('Train.csv','rU'),dialect=csv.excel)
testSize=1001
nsamples=6034195
X_test, y_test = getBatch(size=testSize,doc_iter=csvfile,
		testMode=True,start=len(taglist)-testSize)

lb=preprocessing.LabelBinarizer()
lb.fit(taglist) #taglist is entire list of tags from all CSV rows!
#gotten as for row in csvfile: taglist.append(..)

y_testBinLB=lb.transform(y_test)
y_testBin=labelizer(y_test,all_classes)

#check: np.array_equal(y_testBin.todense(),y_testBinLB)

y_testBinNDArray=np.asarray(y_testBin.todense(),dtype='int')

#to get labels back
y_testRec=lb.inverse_transform(y_testBinNDArray)
#to check this matches original
#must doe fore each sample, e.g. row
#set(y_test[-2]).symmetric_difference(set(y_testRec[-2]))
batchSizeTrain=100
batch_iterator=batchGen(size=batchSizeTrain,doc_iter=csvfile)


#Define a dictionary of stats to keep track off
#to rerun if failed, must do the following three steps
#clf==SGDClassifier() 
#batch_iterator=batchGen(size=batchSizeTrain)
#stats={'n_train':0, 'accuracy': 0.0}

stats={'n_train':0, 'accuracySGD': 0.0, 'accuracyNB': 0.0}

Yfull=labelizer(taglist,classes)

#for non-loopy tests
#X_train, y_train=getBatch(size=900,doc_iter=csvfile)
y_trainSub=np.asarray(y_train[:,samp1000].todense(),dtype='int')
clf.fit(X_train,y_trainSub)



#main loop: iterate on batches
for i, (X_train, y_train) in enumerate(batch_iterator):
	#partial_fit does NOT exist for OVR classifier
	#clf.partial_fit(X_train,y_train,classes=all_classes)
	#densify and make nd.array before fitting
	y_trainDense=np.asarray(y_train.todense())
	clf.fit(X_train,y_trainDense)

	#clf2.partial_fit(X_train,y_train,classes=np.array([0,1]))
	h_pred=clf.predict(X_test)
	#y_pred=uncompressLabel(h_pred) #to be defined
	y_pred=h_pred*Ysub.transpose()*Yfull

	#quik check -label distribution across test samples
	y_pred.sum(1)
	y_testBinLB.sum(1)

	score=metrics.f1_score(y_testBinNDArray,y_pred)

	stats['n_train'] += X_train.shape[0]
	#stats['accuracySGD'] = clf.score(X_test,y_test)
	stats['accuracyNB'] = score	
	if i % 10 == 0:
		print stats


		

