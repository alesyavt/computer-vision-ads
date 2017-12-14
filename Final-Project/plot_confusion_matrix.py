'''
    Authors -
    Alesya Trubchik and Divya Agarwal
'''

# -*- coding: utf-8 -*-
# Import the used libraries
import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

y_true = np.load('y-true-final.pkl')

y_pred = np.load('y-pred-final.pkl')

print y_true
print y_pred

print 'y_true', y_true.shape
print 'y_pred', y_pred.shape



print np.argwhere(y_true == 37)
print np.argwhere(y_pred == 37)

print metrics.classification_report(y_true, y_pred)
print metrics.accuracy_score(y_true, y_pred)

confusion = metrics.confusion_matrix(y_true, y_pred,np.arange(39))

print np.max(confusion)

diag = np.diag(confusion)

print np.max(diag)

print 'sum', np.sum(confusion)
min_ind = diag.argsort()[:10]
print '\npoorest performing class numbers::', min_ind

max_ind = diag.argsort()[-5:]
print '\nBest performing class numbers::', max_ind


plt.imshow(confusion, cmap='gray',vmax =80)
plt.colorbar()
class_numbers = np.arange(0,39,2)
plt.xticks(class_numbers)
plt.yticks(class_numbers)
plt.title('Confusion matrix')

plt.show()
