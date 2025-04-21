#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


import sys
#print(sys.version)

import warnings
warnings.filterwarnings('ignore')


from sklearn.neural_network import MLPClassifier
import csv




from sklearn import preprocessing
import numpy as np

csv_training_data="/home/behnam/Desktop/testData4.csv"
lables=[]
samples=[]
with open(csv_training_data,'r') as csv_file:
    reader = csv.DictReader(csv_file,delimiter=';')
    for row in reader:
        sample=[float(row['a']) , float(row['b']) , float(row['c']) , float(row['d']) , float(row['e']) ]
        samples.append(sample)
        lables.append(int(row['t']))

#print max(lables)
#print min(lables)



samples = preprocessing.scale(samples)

training_set=samples[0: int( 0.9* len(samples) )  ]
test_set=samples[int( 0.9* len(samples) ): len(samples)  ]

training_set_lables=lables[0: int( 0.9* len(samples) )  ]
test_set_lables=lables[int( 0.9* len(samples) ): len(samples)  ]

#print training_set
#print test_set

#print training_set_lables
#print test_set_lables

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 9), random_state=1)
clf.fit(training_set, training_set_lables)
#clf.

#X = [[0., 0.], [1., 1.]]
#y = [0, 1]
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

#clf.fit(X, y) 

#print clf.predict([[2., 2.], [-1., -2.]])

predicted_lables=clf.predict(test_set)
#print predicted_lables 
#print test_set_lables
correct=0
incorrect=0
for i in np.arange(len(test_set_lables)):
    #print i
    if predicted_lables[i]==test_set_lables[i] :
        #print 'correct'
        correct=correct+1
    else:
        #print 'incorrect'
        incorrect=incorrect+1
print correct
print float(correct)/len(test_set_lables)



