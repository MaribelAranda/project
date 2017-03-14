import sys
#Extract data and organized in 3 lists (ID, seq, feature=label)

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


#print(title)
#print(seq)
#print(feature)



#########################################################################################################

#List of amino acids that are present in my sequences (in this case, it resulted that there is an extra amino acid called'X')

n= set()
for line in seq:
        
    for letter in line:
        n.add(letter)
  
#print(list(n))


########################################################################################################

#Assign a number for each amino acid to fit the requirements for an input in svm

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

#Assign a vector position to each amino acid 

list_position=list()
sequence = list()
for serie_amino in aa_num:
	    
    for num_amino in serie_amino:
        position=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        position[num_amino-1]=1
                           
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





 
#######################################################################################################
       
# Assign a number to each label/feature

label_dict= dict(M=0, I=1, O=1, G=2)
label_num= list()
for state in feature:
    for label in state:
        if label in label_dict:
            label_num.append(label_dict[label])
#print(label_num)

#list1=[]
#for element in label_num:
    
 #   if element==0:
  #      list1.append(element)
   #     print(len(list1))
    
            

#######################################################################################################

# Training SVM with the created input  (transform in matrices and see if it gives a score, what means that SVM accept the created input)
     
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
X=np.array(windows)
Y=np.array(label_num)
X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#clf=svm.LinearSVC(class_weight={0: 4.7, 1:95, 2:0.30})
clf=svm.LinearSVC(class_weight='balanced')
clf = clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
print(clf.score(X_test,Y_test))


#######################################################################################################

# Change weights of features

import numpy as np
from sklearn.externals import six
from sklearn.utils.fixes import in1d
from sklearn.utils.fixes import bincount

classes = np.array([0,1,2])
class_weight='balanced'
class_weight={0: 0.2846, 1:0.14418, 2:0.003453} 



def compute_class_weight(class_weight, classes, Y):

    from sklearn.preprocessing import LabelEncoder

    if set(Y) - set(classes):
        raise ValueError
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        Y_ind = le.fit_transform(Y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(Y) / (len(le.classes_) *
                               bincount(Y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]

    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight
#print(compute_class_weight(class_weight, classes, Y))





        
#######################################################################################################

# Cross-validation

from sklearn.model_selection import cross_val_score
#clf = svm.LinearSVC(class_weight={0: 28.46, 1:14.418, 2:0.3453}, C=1)
clf = svm.LinearSVC(class_weight='balanced', C=1)
scores= cross_val_score(clf, X, Y, cv=5) 
print(scores)



# Confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#class_weights= compute_class_weight('balanced', np.unique(Y_train), Y_train) 
#target_names=['class 0', 'class 1', 'class 2']  
   
clf=svm.LinearSVC(class_weight='balanced')
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
print (confusion_matrix(Y_test, predicted))


# Plot confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
classes=[0,1,2]
#clf=svm.LinearSVC(class_weight='balanced')
#clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
cmap=plt.cm.Blues
cm = confusion_matrix(Y_test, predicted)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Prediction 3 states, whole dataset (3 wsize)")
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

