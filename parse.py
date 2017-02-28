def parse(lt):
	lt_nopunc=[(a,b) for (a,b) in lt if (a not in string.punctuation,
					     b not in string.punctuation)]
	lt_lower=[(a.lower(),b.lower()) for (a,b) in lt_nopunc]
	#keep only non("^") non-alphanumeric ("\W") and digits ("\d") in doc
	lt_onlyWords=[(a,b) for (a,b) in lt_lower if re.findall(r"[^\W\d]",b)]

	#remove stopwords
	stopwords=stopwords.words('english')
	stopwords.append('said')
	lt_sw=[(a,b) for (a,b) in lt_onlyWords if b not in stopwords]
	#stem
	porter=nltk.PorterStemmer()
	lt_stem=[(a,porter.stem(b)) for (a,b) in lt_sw]

	return lt_stem
