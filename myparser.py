def myparser(s):
	punc='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n ]'
	np=[a for a in re.split(punc,s) if a not in string.punctuation]
	low=[a.lower() for a in np if len(a)>2]
	nostop=[a for a in low if a not in stopwords.words('english')]
	return [porter.stem(a) for a in nostop if re.findall(r"[^\W\d]",a)]
