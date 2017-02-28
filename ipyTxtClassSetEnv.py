import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
import re
import string
import random


#-------------------------------------
#Exploratory analysis to pick features
#-------------------------------------


#create doc
#create list of tuples cat_word=(tag,document)
#cat_word=[(cat,word)
#for cat in reuters.categories()
#for word in reuters.words(categories=cat)]


def parse(lt):
	lt_nopunc=[(a,b) for (a,b) in lt if (a not in string.punctuation,
					     b not in string.punctuation)]
	lt_lower=[(a.lower(),b.lower()) for (a,b) in lt_nopunc]
	#keep only non("^") non-alphanumeric ("\W") and digits ("\d") in doc
	lt_onlyWords=[(a,b) for (a,b) in lt_lower if re.findall(r"[^\W\d]",b)]

	#remove stopwords
	sw=stopwords.words('english')
	#we noticed 'said' occured too much so we added it to sw
	sw.append('said')
	lt_sw=[(a,b) for (a,b) in lt_onlyWords if b not in sw]
	#stem
	porter=nltk.PorterStemmer()
	lt_stem=[(a,porter.stem(b)) for (a,b) in lt_sw]

	return lt_stem

tag_word=parse(cat_word)

#create cfd
cfd=nltk.ConditionalFreqDist(tag_word)
#make a list of lists of the topX words as features

topX=5
feat=[]
for condition in cfd.conditions():
    feat.append((cfd[condition].keys()[:topX],condition))

#we can flatten the topX lists above and get unique elements
flat_feat=list(unique([y for (a,b) in feat for y in a]))
#get rid of 1 or 2 letter strings
flat_feat_pruned=[a for a in flat_feat if len(a)>2]
#note that once train and test are define below
#we can also use the following to recover flat_feat_pruned
flat_feat_pruned=test[0][0].keys()
#and can replace test with train and any int in range for first entry

#boolean feature vector
def doc_features(doc):
	doc_words=set(doc)
	features={}
	for word in flat_feat_pruned:
		features['%s' % word]=word in doc
	return features
#count based feature vector 
def count_features(doc):
	doc_words=set(doc)
	features={}
	for word in flat_feat_pruned:
		features['%s' % word]=doc.count(word)
	return features

#full_feat=[(doc_features(cfd[condition].keys()),condition=='gold') for condition in cfd.conditions()]

#-------------------------------------
#Use chosen features to create train/test sets
#either based on boolean or count of features in docs
#-------------------------------------


#reutDocs=[(list(reuters.words(fileid)),cat)
#better way: lower, stem, and only keep alpha numeric
reutDocs=[([porter.stem(a.lower()) for a in reuters.words(fileid) if re.findall(r"[^\W\d]",a)] ,cat)
for cat in reuters.categories()
for fileid in reuters.fileids(cat)]

#randomize for later test/train split
random.seed(1979)
random.shuffle(reutDocs)



#run feature extractor on reutDocs instead of cfd; pickled as 'featureset.pkl'
#Boolean based features e.g. 1 or 0 for each word in flat_feat_pruned per document
featureset=[(doc_features(d),c) for (d,c) in reutDocs]
#Count based features
count_featureset=[(count_features(d),c) for (d,c) in reutDocs]

#-------------------------------------
#START from here for classifier algo analysis
#just unpickle featureset and count_featureset
#-------------------------------------


#in future unpickle as 
#featureset=pickle.load(open('featureset.pkl','rb'))
#count_featureset=pickle.load(open('count_featureset.pkl','rb'))



#get a sense of how often features 'flat_feat_pruned' appear in each doc
tmp=[sum(featureset[k][0].values()) for k in range(len(featureset))]
hist(tmp)

#size of train test split (90,10)%
size=int(len(featureset)*0.1)
train=featureset[size:]
test=featureset[:size]

train_count=count_featureset[size:]
test_count=count_featureset[:size]

#-------------------------------------
#To run classifier natively in sklearn
#-------------------------------------
from sklearn.feature_extraction import DictVectorizer
dicvec=DictVectorizer()
#tdicvec=DictVectorizer() #BAD--don't creat new object for test
			  #use train object and .transform() for test

#get just the features (not tags) from count based train set
#will work with boolean too, but won't work later with tf-idf transformer 
train_feat=[a[0] for a in train]
test_feat=[a[0] for a in test]

#test and train features
dicvec_feat=dicvec.fit_transform(train_feat)
tdicvec_feat=dicvec.transform(test_feat)

from sklearn.feature_extraction.text import TfidfTransformer
tfxfr=TfidfTransformer()
tfidf_feat=tfxfr.fit_transform(dicvec_feat)
test_tfidf_feat=tfxfr.fit_transform(tdicvec_feat)

#create tag list
train_tag=[a[1] for a in train]
test_tag=[a[1] for a in test]

#label encode test tags as integers
from sklearn import preprocessing


#define classifier and train
from sklearn.svm import LinearSVC

clf=LinearSVC()



#not better but perhaps??
le=preprocessing.LabelEncoder()
le.fit(train_tag) #or le.fit(reuters.categories())
clf.fit(dicvec_feat,le.transform(train_tag))
pred_train=clf.predict(dicvec_feat)
tmp=(pred_train==le.transform(train_tag))
tmp.sum()
pred=clf.predict(tdicvec_feat)
tmp_test=(pred==le.transform(test_tag))
tmp_test.sum()

from sklearn import metrics

score=metrics.f1_score(test_tag,pred)

cm_sk=nltk.ConfusionMatrix(test_tag,le.inverse_transform(pred).tolist())
print cm_sk.pp(sort_by_count=True, show_percents=False, truncate=5)

#-------------------------------------
#To run classifier with nltk scikit-interface
#-------------------------------------

#setup and train a classifier
cf=nltk.NaiveBayesClassifier.train(train)
#show some discrimination power
cf.show_most_informative_features(5)
#test the cf and show accuracy
print nltk.classify.accuracy(cf,test) #0.6

#create a confusion matrix
#note percentages on diags is number of times correct label is applied/size of test set
#not percent of times label is applied correctly
pred_NB=cf.batch_classify(test_feat)
#results=[cf.classify(test[a][0]) for a in range(size)]
#gold=[test[a][1] for a in range(size)]
cm_NB=nltk.ConfusionMatrix(test_tag,pred_NB)
print cm_NB.pp(sort_by_count=True, show_percents=False, truncate=10)

#create structures for classification
test_doc=[a[0] for a in test]

#build, train, and test classifiers
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
sv=SklearnClassifier(LinearSVC())
sv.train(train)
#note that train performance matches tmp.sum()
pred_train_sv=sv.batch_classify(train_feat)
nltk.ConfusionMatrix(train_tag,pred_train_sv)
#also test performance matches tmp_test.sum()
pred_sv=sv.batch_classify(test_feat)
#confusion matrices
cmsv=nltk.ConfusionMatrix(test_tag,pred_sv)
print cmsv.pp(sort_by_count=True, show_percents=False, truncate=5)
#some SklearnClassifier internals
featsets, labs = zip(*train)
X = sv._convert(featsets)
import numpy
y=numpy.array([sv._label_index[l] for l in labs])
#then to train one would use sv._clf.fit(X,y)

#-------------------------------------
#To vectorize/classify all in sklearn
#-------------------------------------
porter=nltk.PorterStemmer()
#for use with sklearn 
def myparser(s):
	punc='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n ]' #all punc+whtspc+newline
	np=[a for a in re.split(punc,s) if a not in string.punctuation]
	low=[a.lower() for a in np if len(a)>2] #only two-lett words lowered
	nostop=[a for a in low if a not in stopwords.words('english')]
	return [porter.stem(a) for a in nostop if re.findall(r"[^\W\d]",a)]
#imports
from sklearn.feature_extraction.text import TfidfVectorizer
#object instantiation - ignore utf-8 decode errors
vectfidf=TfidfVectorizer(tokenizer=myparser,decode_error='ignore')
#test corpus
#corpus=[reuters.raw('training/9853'),reuters.raw('training/9866')]

#reuters corpus
corpus=[(reuters.raw(fileid),cat) for cat in reuters.categories()
for fileid in reuters.fileids(cat)]
random.seed(1979)
random.shuffle(corpus)
size=int(len(corpus)*0.1)
train_raw=corpus[size:]
test_raw=corpus[:size]
train_raw_data=[a[0] for a in train_raw]
test_raw_data=[a[0] for a in test_raw]


y_train=[a[1] for a in train_raw]
y_test=[a[1] for a in test_raw]
#DO NOT have to turn labels to ints
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(reuters.categories())

#get all features
X_train=vectfidf.fit_transform(train_raw_data)
X_test=vectfidf.transform(test_raw_data)
#get chi-2 features only
from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=1000) #selecting k too small, e.g. 100, gives 'Dup scores' warning
X_train_ch2=ch2.fit_transform(X_train, le.transform(y_train))
X_test_ch2=ch2.transform(X_test)
clf=LinearSVC()
clf.fit(X_train_ch2,le.transform(y_train))
pred=clf.predict(X_test_ch2)

from sklearn import metrics
score=metrics.f1_score(le.transform(y_test),pred)

#can try using SGD
clf2=SGDClassifier()
clf2.fit(X_train_ch2,le.transform(y_train))
pred2=clf2.predict(X_test_ch2)
score2=metrics.f1_score(le.transform(y_test),pred2)




#----------------------------
#20 news group corpus
#----------------------------


#to load 20newsgroup dataset in sklearn w/o fetching
from sklearn.datasets import load_files
train=load_files('/Users/ashah/scikit_learn_data/20news_home/20news-bydate-train/', encoding='utf-8', decode_error='ignore')

test=load_files('/Users/ashah/scikit_learn_data/20news_home/20news-bydate-test/', encoding='utf-8', decode_error='ignore')

from pprint import pprint
pprint(list(train.target_names))

#output
y_train=train.target
#extract input/predictors
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(train.data)
X_train.get_shape() #rows match len(y_train) or y_train.shape

X_test = vectorizer.transform(test.data)

#select chi2
from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=1000) #selecting k too small, e.g. 100, gives 'Dup scores' warning
X_train_ch2=ch2.fit_transform(X_train, y_train)
X_test_ch2=ch2.transform(X_test)

#build a classifier
clf=LinearSVC()
clf.fit(X_train_ch2,y_train)
pred=clf.predict(X_test_ch2)

from sklearn import metrics
score=metrics.f1_score(y_test,pred)
print metrics.classification_report(y_test,pred,target_names=test.target_names)

