-----------------------------------------------------------------
import ntlk

from nltk.corpus import reuters



#---parse routine---
import string

#do for each tag
#get words in doc for each tag
goldWords=reuters.words(categories=['gold'])

#remove punctuations from words in doc
goldWords_nopunc=[word for word in goldWords if word not in string.punctuation]
#lowercase
goldWords_nopunc=[x.lower() for x in goldWords_nopunc]

#keep only non("^") non-alphanumeric ("\W") and digits ("\d")
import re
goldWords_nopunc_onlyWords=[w for w in goldWords_nopunc if re.findall(r"[^\W\d]",w)]

#remove stopwords
from nltk.corpus import stopwords
stopwords=nltk.corpus.stopwords.words('english')
tmp=[w for w in goldWords_nopunc_onlyWords if w not in stopwords]

#stem
porter=nltk.PorterStemmer()
tmp2=[porter.stem(w) for w in tmp]

#freq distribution
goldFreq=nltk.FreqDist(tmp2)
goldWords_mostFreq=[w for w in goldFreq.keys() if goldFreq[w]>50]

#cfd after features have been extracted
cat_word=[(cat,word)
for cat in reuters.categories()
for word in reuters.words(categories=cat)]

cfd=nltk.ConditionalFreqDist(cat_word)
cfd.conditions()
cfd['gold']
list(cfd['gold'])
reuters.words(categories='gold')

#make a list of lists of the topX words as features

topX=5
feat=[]
for condition in cfd.conditions():
    feat.append(cfd[condition].keys()[:topX])
	

