28c28
< from sklearn.metrics import hamming_loss, f1_score #accuracy_score
---
> from sklearn.metrics import hamming_loss #accuracy_score
31d30
< from sklearn.preprocessing import LabelBinarizer
90d88
< 		#return hasher.transform(X), lb.transform(y,all_classes)
99,113d96
< #Define a Testset batch getter from csv file reader object 
< def getBatchTestSet(size,doc_iter,testMode=False,start=0):
< 	data=[(doc[0],doc[2]) for doc in 
< 			itertools.islice(doc_iter,start,start+size)]
< 	iden,X=zip(*data)
< 	return hasher.transform(X), iden
< 
< #Define a Test generator using the above batch getter
< def batchGenTestSet(size,doc_iter=None):
< 	X, iden=getBatchTestSet(size,doc_iter) #need to do this to def stop condition
< 	while X.shape[0]: #stop condition here
< 		yield X,iden #yield keyword defines a generator
< 		X, iden=getBatchTestSet(size,doc_iter)
< 
< 
120,125c103
< clf=SGDClassifier()
< #Load column indices of label matrix that we'll use for training
< #samp1000=pickle.load(open('samp1000.pkl','rb'))
< samp=pickle.load(open('samp1500.pkl','rb'))
< #Load corresponding label projection matrix
< proj=pickle.load(open('proj1500.pkl','rb')) #add switch for 100/1000
---
> clf=SGDClassifier(n_jobs=10)
132,133d109
< #To use LabelBinarizer(), we need the entire
< #training set ensure all classes are known
135,136d110
< 
< 
141c115
< csvfile=csv.reader(open('subTrain.csv','rU'),dialect=csv.excel)
---
> csvfile=csv.reader(open('Train.csv','rU'),dialect=csv.excel)
151,156c125,126
< stats={'n_train':0, 'accuracyHL': 0.0, 'F1':0.0, 'accuracy_history': [(0, 0,0)]
< 		, 't0': time.time(),'runtime_history': [(0, 0)]}
< 
< #To proceed without looping (for dev purposes)
< #ebi=enumerate(batch_iterator)
< #i, (X_train,y_train)=ebi.next()
---
> stats={'n_train':0, 'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
>          'runtime_history': [(0, 0)]}
161c131
< 	y_trainSub=np.asarray(y_train[:,samp].todense(),dtype='int')
---
> 	y_trainSub=np.asarray(y_train.todense(),dtype='int')
167,168d136
< 	#elif i > 5: # five loops for testing only
< 		#break
176,183d143
< 	#Project the prediction to the full label space
< 	y_pred=proj.dot(preds)
< 	#Round to either 0 or 1 for each entry (decision)
< 	y_predRnd=(y_pred.transpose()>0.5).astype(int)
< 	#To get actual class labels in a list-of-lists (list for each row)
< 	#lolPred=[all_classes[np.ravel(np.nonzero(y_predRnd[k,:]))].tolist() 
< 		#	for k in range(y_predRnd.shape[0))]
< 	
187,220c147,150
< 	stats['accuracyHL']=hamming_loss(y_testBin,y_predRnd)*42048
< 	stats['F1']=f1_score(y_testBin,y_predRnd)
< 	stats['accuracy_history'].append((stats['accuracyHL'],
< 		stats['F1'], stats['n_train']))
< 	stats['runtime_history'].append((stats['accuracyHL'],time.time() - stats['t0']))
< 	print stats['n_train'], stats['accuracyHL'], stats['F1']
< 
< 
< #Now run to predict labels on the test set
< testSet=csv.reader(open('Test.csv','rU'),dialect=csv.excel)
< batchSizeTest=100
< batch_iteratorTest=batchGenTestSet(size=batchSizeTest,doc_iter=testSet)
< lolPred=[] #List-of-Lists predicted multi-label output
< idenList=[]
< for i, (X_testSet,iden) in enumerate(batch_iteratorTest):
< 	preds=np.array([np.ravel(e.predict(X_testSet)) for e in ests])
< 	#Project the prediction to the full label space
< 	y_pred=proj.dot(preds)
< 	#Round to either 0 or 1 for each entry (decision)
< 	y_predRnd=(y_pred.transpose()>0.5).astype(int)
< 	#To get actual class labels in a list-of-lists (list for each row)
< 	lolPred += [all_classes[np.ravel(np.nonzero(y_predRnd[k,:]))].tolist() 
< 			for k in range(y_predRnd.shape[0])]
< 	idenList += iden
< 
< #Write the predicted labels to a file
< fo=open('predFB.csv','wb')
< wr=csv.writer(fo)
< wr.writerow(['Id','Tags']) #header
< for i in range(len(lolPred)):
< 	wr.writerow([idenList[i],str(' '.join(lolPred[i]))])
< 
< fo.close()
< 
---
> 	stats['accuracy']=hamming_loss(y_testBin,preds.T)*42048
> 	stats['accuracy_history'].append((stats['accuracy'], stats['n_train']))
> 	stats['runtime_history'].append((stats['accuracy'],time.time() - stats['t0']))
> 	print stats['n_train'], stats['accuracy']
226a157,158
> #Now use our model to predict on the actual test set
> #testfile=csv.reader(open(..
