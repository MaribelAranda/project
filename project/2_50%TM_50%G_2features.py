# Create a new dataset with 50%globular proteins and 50% TM to try to distinguish and predict transmembrane and globular.

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
#print(len(tm))

while len(ntm) < len(tm):
    if total[g].startswith('G'):
        ntm.append(total[g].strip('\n'))
        ntm.append(total[g-1].strip('\n'))
    g = g + 3
#print(ntm)
#print(len(ntm))


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
 
#Assign a number for each amino acid to fulfil requirements for an input for svm


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

#Assign a vector to each amino acid 

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

#######################################################################################################
       
# Assign a number to each label/feature

label_dict= dict(M=0, I=1, O=1, G=1)
label_num= list()
for state in half_label:
    for label in state:
        if label in label_dict:
            label_num.append(label_dict[label])
#print(label_num)


        
#######################################################################################################

# Creating matrices as input for SVM  
     
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
X=np.array(windows)
Y=np.array(label_num)
X.shape


#######################################################################################################

# Cross-validation

from sklearn.model_selection import cross_val_score
#clf = svm.LinearSVC(class_weight={0: 28.46, 1:14.418, 2:0.3453}, C=1)
clf = svm.LinearSVC(class_weight='balanced', C=1)
scores= cross_val_score(clf, X, Y, cv=5) 
#print(scores)



# Training SVM

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = clf.fit(X_train, Y_train)
#print(clf.score(X_test,Y_test))





# Confusion matrix

import numpy as np
from sklearn.metrics import confusion_matrix
   
#clf=svm.LinearSVC(class_weight={0:1.26, 1:0.82})
clf=svm.LinearSVC(class_weight='balanced')
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
#print (confusion_matrix(Y_test, predicted))





#######################################################################################################

# Changing weights of features


from sklearn.externals import six
from sklearn.utils.fixes import in1d
from sklearn.utils.fixes import bincount
classes = np.array([0,1])
class_weight='balanced'


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



