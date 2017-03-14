#Extract data and organized in 3 lists (ID, seq, feature=label)

import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
f = open('beta_globular_project', 'r')


total=list(f)
title=list()
seq=list()
feature=list()
for line in range(0, len(total), 3):    
    title.append(total[line].strip('\n'))   
for line in range(1, len(total), 3):
    seq.append(total[line].strip('\n'))   
for line in range(2, len(total), 3):
    feature.append(total[line].strip('\n'))



############################################################################################################################

## Relate whole list of ID with the few ID I have in pssm and transform data from pssm to convert it in an input for svm


vectors=[]
feature_new=[]
wsize= int(input('Enter odd number as a window size: '))
windows= list()
pad = (wsize - 1) // 2
#null_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
null_pad = [0.0]*20



for id_ in title:
    newid_ = id_.strip('>')
    
    if os.path.isfile('./pssm_tm_g/%s.fasta.pssm'%newid_):
        
        #print ("Opening",newid_)
        o=open('./pssm_tm_g/%s.fasta.pssm'%newid_)
        #print('%s' %newid_)
        contents = o.readlines()
        perlines =[]
        for line in contents[3:len(contents)-6]:
        #print(line)
            
            perline= line.split() # Each line is a list        
            perline= perline[22:42]
            perline= [int(i) for i in perline]
            perlines.append(perline)
        #print(perlines)
        for number in perlines:
            for index in range(len(number)):
                number[index]= number[index]/100
        for aaindex in range(len(perlines)):
            
            
            if aaindex==0:   
                windows.append(null_pad*pad + perlines[aaindex] + perlines[aaindex + 1]*pad)

            elif aaindex==(len(perlines)-1):
                windows.append(perlines[aaindex-1]*pad + perlines[aaindex] + null_pad*pad)
            else:
                windows.append(perlines[aaindex-1]*pad + perlines[aaindex] + perlines[aaindex+1]*pad)
        X=np.array(windows)

        #Relate whole list of ID with the features that correspond with the few sequences I run in PSI_BLAST. Thereby, sequences and features are connected through their ID 
        index= title.index(id_)
        feature_new.append(feature[index])

        label_dict= dict(M=0, I=1, O=1, G=2)
        label_num= list()
        for state in feature_new:
            for label in state:
                if label in label_dict:
                    label_num.append(label_dict[label])
        Y=np.array(label_num)
#print(label_num)
        
#print(windows)
      

############################################################################################################################



### Cross-validation
from sklearn.model_selection import cross_val_score
#clf = svm.LinearSVC(class_weight={0: 28.46, 1:14.418, 2:0.3453}, C=1)
clf = svm.LinearSVC(class_weight='balanced', C=1)
scores= cross_val_score(clf, X, Y, cv=5) 
print(scores)

#train SVM
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = svm.LinearSVC(class_weight='balanced', C=1)
clf = clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))       

  
# Confusion matrix

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
  
#clf=svm.LinearSVC(class_weight={0:1.26, 1:0.82})
predicted = clf.predict(X_test)
print (confusion_matrix(Y_test, predicted))      
#sys.exit()          
    

       
# Plot confusion matrix
classes=[0,1,2]
#clf=svm.LinearSVC(class_weight='balanced')
#clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
cmap=plt.cm.Blues
cm = confusion_matrix(Y_test, predicted)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("TM - 225G pssm_model(11wsize)")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)    

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, round(cm[i, j],5),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()       
                      
            
import pickle   
s= pickle.dumps(clf)





