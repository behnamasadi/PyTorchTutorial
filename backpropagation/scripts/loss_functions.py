import  matplotlib.pyplot as plt
import numpy as np

def hing_loss(classification_multiplied_by_label):
        return np.maximum(0,1-classification_multiplied_by_label)


def zero_one_loss(classification_multiplied_by_label):
        return [ 0 if x>0 else 1   for x in classification_multiplied_by_label]

def logistic_loss(classification_multiplied_by_label):
        return np.log(1+np.exp(-classification_multiplied_by_label))

def adaboost(classification_multiplied_by_label):
        return np.exp(-classification_multiplied_by_label)

classification_multiplied_by_label=np.linspace(-2,2,200)

hing_loss_output=hing_loss(classification_multiplied_by_label)
zero_one_loss_output=zero_one_loss(classification_multiplied_by_label)
logistic_loss_output=logistic_loss(classification_multiplied_by_label)
adaboost_output= adaboost(classification_multiplied_by_label)


hing_loss_label,=plt.plot(classification_multiplied_by_label,hing_loss_output, label='hing (SVM)')
zero_one_label,=plt.plot(classification_multiplied_by_label,zero_one_loss_output, label='zero one')
logistic_loss_label,=plt.plot(classification_multiplied_by_label,logistic_loss_output, label='logistic')
adaboost_loss_label,=plt.plot(classification_multiplied_by_label,adaboost_output, label='adaboost')

plt.legend(handles=[hing_loss_label,zero_one_label,logistic_loss_label,adaboost_loss_label])

plt.grid(True, which='both')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.show()
