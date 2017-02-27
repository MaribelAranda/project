#Extract data and organized in 3 lists (ID, seq, feature=label)
f = open('sample_30', 'r')

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

#for line in f:    
 #   if line.startswith ('>'):
  #      title.append(line)
   # elif line.startswith ('M'):        
    #    seq.append(line)
    #if not line.startswith ('>') and not line.startswith('M'):
     #   feature.append(line)

        
#print(title)
#print(seq)
#print(feature)

#########################################################################################################

#List of aa that are present in my sequences
n= set()
for line in seq:
        
    for letter in line:
        n.add(letter)
  
#print(list(n))


########################################################################################################

#Assign a number for each aa since svm cannot read letters

#never use mapfunction as variable 
#map= {'A':1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,  'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20, 'X':21}

aa=dict(A=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, K=9, L=10, M=11, N=12, P=13, Q=14, R=15, S=16, T=17, V=18, W=19, Y=20, X=21)

aa_num=list()
for sequence in seq:
    temp = []
    for letter in sequence:
        if letter in aa:
            temp.append(aa[letter])
    aa_num.append(temp)
#print(aa_num[:3])


########################################################################################################

#Assign a position to each aa and add two more position vectors without any aminoacid in order to enable the first and last aa to be in the middle of a window

list_position=list()
for serie_amino in aa_num:
    #temp2= []
    for num_amino in serie_amino:
        position=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        position[num_amino-1]=1
        #temp2.append(position)
        #list_position.append(temp2)
        list_position.append(position)
        
#temp2.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#null_vector=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
#vector_position = null_vector + temp2
#print(list_position[:5])

#wsize= int(input('Enter odd number as a window size: '))
#window= list()
#pad = (wsize - 1) // 2
#null_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#for sequence in list_position:
    
 #   sequence = [null_pad] * pad + sequence + [null_pad] 
  #  for index in range(len(sequence)):
     #   b=[]
      #  for e in sequence:
       #     b.extend(e)
   #    window.append(sequence[index:index + wsize])
        #    window.append(b)
                #output2.write(window.append(b))
#print(window[:8])


#for number in aa_num:
#position_dict={1:'100000000000000000000', 2:'010000000000000000000', 3:'001000000000000000000', 4:'000100000000000000000', 5:'000010000000000000000', 6:'000001000000000000000', 7:'000000100000000000000', 8:'000000010000000000000', 9:'000000001000000000000', 10:'000000000100000000000', 11:'000000000010000000000', 12:'000000000001000000000', 13:'000000000000100000000', 14:'000000000000010000000', 15:'000000000000001000000', 16:'000000000000000100000', 17:'000000000000000010000', 18:'000000000000000001000', 19:'000000000000000000100', 20:'000000000000000000010', 21:'000000000000000000001'}



#######################################################################################################

#Create windows 

#wsize= int(input('Enter odd number as a window size: '))
#window= list()

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



  #  for sequence in list_position:
   #     for window in sequence:
            
    #        window.append(list_position[sequence][window:window+(wsize)])






#for sequences in range(0, len(list_position)):
 #   for single_seq in range(0, (len(list_position[sequences]))):
  #      for vector in range(0, (len(list_position[sequences[single_seq]])-(wsize-1))):
   #         window.append(list_position[sequences[single_seq]][vector:vector+(wsize)])   
#print(window)     
  

     
#for sequence in range(0, len(aa_num)):
 #   for letter in range(0, (len(aa_num[sequence])-(wsize-1))):
  #      window.append(aa_num[sequence][letter:letter+(wsize)])
#list_wsize=list()
#random_list=[[0,0,0,1], [0,0,1], [1,0,0,0,0], [0,1,0,0,0]]
#random_list.append([0,0])

#import numpy as np
#window_list=list()
#final_vector= [[0], [0,0,0,1], [0,0,1], [1,0,0,0,0], [0,1,0,0,0], [0,0]]
#for vector in range(0, len(final_vector)):
 #   window_list.append(''.join(vector[:3]))

	#window_list.append(vector)
#print(final_vector)
#for vector in final_vector:
    #array_pos=np.array(final_vector[:3])
#print(window_list)



#	aa_num=[map[aa_map] for aa_map in item] 



#for num in range(0, len(aa_num)):
 #   position=[map[char]for char in num]
  #  print(position) 
#   vector = position+1
#print(vector) 
    #if aa_num == '11':
      #  position2=float.position[1]+1
    #sum_1=list()
    #sum_1.append(aa_num[amino]+1)
 
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

# Check input for SVM        
import numpy as np
X=np.array(list_position)
Y=np.array(label_num)
#print(Y)
from sklearn import svm           
clf=svm.SVC().fit(X,Y)
print(clf.score(X,Y))
        


#import numpy as np
#a = np.array([[[title],[seq],[feature]]])
#fr
