# # Create a new dataset with approximate the same number of labels belonging to a transmembrane and a globular protein (extraction of all transmembrane and 225 globular proteins from the original dataset)

f = open('beta_globular_project', 'r')

total=list(f)
tm=list()
ntm=list()
i=2
d=2
m=2
g=2

for line in range(2, len(total), 3):
    if total[i].startswith('I'):
        tm.append(total[i].strip('\n'))
        tm.append(total[i-1].strip('\n'))
    i = i + 3

for line in range(2, len(total), 3):
    if total[d].startswith('O'):        
        tm.append(total[d].strip('\n'))
        tm.append(total[d-1].strip('\n'))
    d = d + 3

for line in range(2, len(total), 3):
    if total[m].startswith('M'):        
        tm.append(total[m].strip('\n'))
        tm.append(total[m-1].strip('\n'))
    m = m + 3

#print(tm)
print(len(tm))

while len(ntm) < 225:
    if total[g].startswith('G'):
        ntm.append(total[g].strip('\n'))
        ntm.append(total[g-1].strip('\n'))
    g = g + 3
#print(ntm)
print(len(ntm))


tm.extend(ntm)
print(len(tm))


half_seq=list()
half_label=list()
for line in range(0, len(tm), 2):    
    half_label.append(tm[line].strip('\n'))   
for line in range(1, len(tm), 2):
    half_seq.append(tm[line].strip('\n')) 
#print(len(half_label))
#print(len(half_seq)) 

##############################################################################################################################################################
 
#Assign a number for each amino acid


aa={'A':1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,  'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20, 'X':21}


aa_num=list()
for sequence in half_seq:
    temp = []
    for letter in sequence:
        if letter in aa:
            temp.append(aa[letter])
    aa_num.append(temp)
#print(aa_num[:3])
#print(len(aa_num))

########################################################################################################

#Assign a vector to each aa 

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
    
    for aaindex in range(len(sequence)):
        
        if aaindex==0:
            windows.append(null_pad*pad + sequence[aaindex] + sequence[aaindex+1]*pad)
            
        elif aaindex==(len(sequence)-1):
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + null_pad*pad)
        else:
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + sequence[aaindex+1]*pad)        
                      
#print(windows[:3])
#print(len(windows))

########################################################################################################


# Assign a number to each label/feature

label_dict= dict(M=0, I=1, O=1, G=2)
label_num= list()
for state in half_label:
    for label in state:
        if label in label_dict:
            label_num.append(label_dict[label])
#print(label_num)

        
#######################################################################################################

# Train random forest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X=np.array(windows)
Y=np.array(label_num)
X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)




##X, Y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)????



                         
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=5, random_state=0, max_features='auto')
clf = clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))



# Confusion matrix

import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split



predicted = clf.predict(X_test)
print (confusion_matrix(Y_test, predicted))
   




# Plot confusion matrix
classes=[0,1,2]
#clf=svm.LinearSVC(class_weight='balanced')
#clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
cmap=plt.cm.Blues
cm = confusion_matrix(Y_test, predicted)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("TM - 225G Random_forest (3features -11wsize)")
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


################################################################################################

# Use of the created predictor model to predict in the whole original dataset

f = open('beta_globular_project', 'r')

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

#List of amino acids that are present in my sequences (in this case, it resulted that there is an extra amino acid called'X')

n= set()
for line in seq:
        
    for letter in line:
        n.add(letter)
  
#print(list(n))


########################################################################################################

#Assign a number for each amino acid


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

#Assign a position to each amino acid 

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
    
    for aaindex in range(len(sequence)):
        
        
        if aaindex==0:
            windows.append(null_pad*pad + sequence[aaindex] + sequence[aaindex+1]*pad)
            
        elif aaindex==(len(sequence)-1):
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + null_pad*pad)
        else:
            windows.append(sequence[aaindex-1]*pad + sequence[aaindex] + sequence[aaindex+1]*pad)
        
       
             
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

    
            

#######################################################################################################

# Load the saved model(clf) and perform a prediction  

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
X=np.array(windows)
Y=np.array(label_num)
X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf2 = pickle.loads(s)
clf2.predict(X_test)
print(clf2.score(X_test, Y_test))



# Confusion matrix

predicted = clf2.predict(X_test)
print (confusion_matrix(Y_test, predicted))



# Plot confusion matrix

import itertools
import matplotlib.pyplot as plt


classes=[0,1,2]
cmap=plt.cm.Blues
cm = confusion_matrix(Y_test, predicted)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Prediction_wholedata_225G_11_wsize)")
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




