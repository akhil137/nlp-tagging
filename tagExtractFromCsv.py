import csv
import pickle
r=csv.reader(open('Train.csv','rU'),dialect=csv.excel)

tagset=set() #create set for unique elements only
taglist=[]

for row in r:
#	tagset |= set(row[3].split(' ')) #union operator faster than add
        taglist.append(row[3].split(' '))

#pickle.dump(tagset,open('tagset.pkl','wb'))
pickle.dump(taglist,open('taglist.pkl','wb'))
