'''
    Authors -
    Alesya Trubchik and Divya Agarwal
'''

import numpy as np
import pickle
 
# Used to make a 80% train / 20% test split on the dataset. Save out indices to be used for test and train
# from 0_stacked_images.pkl

#img_data = np.load('../preprocessed-vgg/0_stacked_images.pkl')
#print img_data.shape
labels = np.load('../labels/0_labels.pkl')
print labels.shape

img_data = labels

train_size = len(img_data)
val_amount = int(train_size * .2)
# choose random indices for test set

mask = np.zeros(train_size, dtype=bool)
val_ind = np.random.choice(train_size, val_amount, replace=False)
mask[val_ind] = True

#train_set = train_data['data'][~mask]
#val_set = train_data['data'][mask]
#print 'train set shape', train_set.shape
#print 'val set shape', val_set.shape

train_ind = np.argwhere(~mask).reshape(-1)
val_ind = np.argwhere(mask).reshape(-1)
print 'train ind set shape', train_ind.shape
print 'val ind set shape', val_ind.shape


pickle.dump(train_ind, open('train-ind.pkl', 'wb'))
pickle.dump(val_ind, open('test-ind.pkl', 'wb'))
