'''
    Authors -
    Alesya Trubchik and Divya Agarwal
'''


import numpy as np
import pickle


# Show image names of correct predictions side by side with their true labels

true = np.load('y-true-final.pkl')
pred = np.load('y-pred-final.pkl')

image_names = np.load('image-index-to-names.pkl')
test_ind = np.load('test-ind.pkl')

test_image_names = image_names[test_ind]

print true
print pred
correct_pred = np.argwhere(true == pred)
print 'true positives'

print 'image names of correct ad classifications and their labels'
combine_correct = np.hstack([test_image_names[correct_pred], true[correct_pred]])


print combine_correct
print 'total correct predictions', len(correct_pred), 'of', len(pred)


print 'image names of INCORRECT ad classifications, their true labels, and predicted labels'

false_pred = np.argwhere(true != pred)
combine_false = np.hstack([test_image_names[false_pred], true[false_pred], pred[false_pred]])
print combine_false
print 'total incorrect predictions', len(false_pred), 'of', len(pred)

