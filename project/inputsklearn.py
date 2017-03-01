import sys
#Extract data and organized in 3 lists (ID, seq, feature=label)
f = open('sample_third', 'r')

#o = open('output_extract', 'w')
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


#print(title)
#print(seq)
#print(feature)



#########################################################################################################

#List of aa that are present in my sequences (in my case, it resulted that I have an extra aa called'X')

n= set()
for line in seq:
        
    for letter in line:
        n.add(letter)
  
#print(list(n))


########################################################################################################

#Assign a number for each aa since svm cannot read letters

#never use mapfunction as variable 
#map= {'A':1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,  'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20, 'X':21}

#aa=dict(A=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, K=9, L=10, M=11, N=12, P=13, Q=14, R=15, S=16, T=17, V=18, W=19, Y=20, X=21)

aa={'A':1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,  'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20, 'X':21}


aa_num=list()
for sequence in seq:
    temp = []
    for letter in sequence:
        if letter in aa:
            temp.append(aa[letter])
    aa_num.append(temp)
#print(aa_num[:3])


########################################################################################################

#Assign a position to each aa 

list_position=list()
sequence = list()
for serie_amino in aa_num:
	
    #temp2= []
    for num_amino in serie_amino:
        position=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        position[num_amino-1]=1
     #   temp2.append(position)
        
      #  list_position.append(temp2)
        
        sequence.append(position)
list_position.append(sequence)        
#print(list_position)

#######################################################################################################

#Create window sizes

wsize= int(input('Enter odd number as a window size: '))
windows= list()
pad = (wsize - 1) // 2
null_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


for sequence in list_position:   
    #print(sequence[1])
    #sequence = null_pad * pad + sequence + null_pad * pad
    #print(len(sequence))
    #break
    for aaindex in range(len(sequence)):
        #print(aaindex)
        #print(sequence[aaindex])
      #  if aaindex==0:
     #       windows.append(null_pad + sequence[aaindex:aaindex+(wsize-1)])
    #    elif (aaindex != 0) and (aaindex != -1):
   #         windows.append(sequence[aaindex:aaindex+wsize])
  #      elif aaindex==len(sequence):
 #           windows.append(sequence[aaindex:aaindex+wsize-1] + null_pad)
#sequence[aaindex-1] + sequence[aaindex] + null_pad)
#sequence[aaindex-1] + sequence[aaindex] + sequence[aaindex+1])        

#print(windows, len(windows))
        
        if aaindex==0:
            windows.append(null_pad*pad + sequence[aaindex] + sequence[aaindex+1]*pad)
            
        elif aaindex==(len(sequence)-1):
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + null_pad*pad)
        else:
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + sequence[aaindex+1]*pad)
        
        #break   
        #sys.exit(10)  
             
#print(windows)
    

        #windows.append(sequence[aa_vector:aa_vector + wsize])        
        



#######################################################################################################


#with open('output.txt', 'w+') as output:
#for protein in list_position:    
 #   for i in range(0, len(protein)):                  
  #      a = protein[i:i+wsize]
        #    output.write(str(a)+'\n')
   #     for vector in a:
    #        b=vector[0] + vector[1] + vector[2] + wsize
        #print(b)

#with open('output2.txt', 'w+') as output2:
 #   for protein in a:
  #      for i in range(0, len(protein)):
   #         b= protein[0] + protein[1] + protein[2]
    #        output2.write(str(b)+'\n')



 
#######################################################################################################
       
# Assign a number to each label/feature

label_dict= dict(M=0, I=1, O=1, G=2)
label_num= list()
for state in feature:
    for label in state:
        if label in label_dict:
            label_num.append(label_dict[label])
#print(label_num)
            

#######################################################################################################

# Check input for SVM  (transform in matrices and see if it gives a score, what means that SVM accept the created input, though it's overfitting)
     
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
X=np.array(windows)
Y=np.array(label_num)
#X.shape
#clf=svm.LinearSVC()
#clf = clf.fit(X,Y)
#print(clf.score(X,Y))

from sklearn.metrics import confusion_matrix
# Split data in training set (80% of total) and test set (20%)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#X_train.shape, Y_train.shape
#X_test.shape, Y_test.shape
        
clf=svm.LinearSVC(class_weight='balanced')
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
print (confusion_matrix(Y_test, predicted))

import sys
sys.exit()

#print(clf.score(X_test,Y_test))


# Confusion matrix
#import itertools
#import numpy as np
#import matplotlib.pyplot as plt

#from sklearn import svm, datasets
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
y_pred= clf.predict(X_test)
confusion_matrix(Y_test, y_pred)

#class_names=[M, L, G]
#classifier = svm.SVC(kernel='linear', C=0.01)
#Y_test = classifier.fit(X_train, Y_train).predict(X_test)
        
#######################################################################################################

# Cross-validation

from sklearn.model_selection import cross_val_score
clf = svm.LinearSVC(class_weight='balanced', C=1)
scores= cross_val_score(clf, X_train, Y_train, cv=5) 
print(scores)

clf = clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))

