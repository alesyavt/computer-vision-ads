import json
import numpy as np
from collections import Counter
import pickle
import os

def get_labels(input_file_path, data, missing_images):

  image_names = os.listdir(input_file_path)
  #print image_names[:10]

  i = 0
  img_names = []
  for img in image_names:
    i += 1
    #if i == 170:
    #  continue
    print img
    img_names.append(subfolder + '/' + img)
    try:
      data[img_names[-1]]
    except:
      missing_images.append(img_names[-1])

  labels = np.zeros((len(img_names),))
  for i in range(len(img_names)):
    try:
      row = Counter(data[img_names[i]])
      mode = row.most_common(1)[0][0]
      if row.most_common(1)[0][1] == 1:
        print row.most_common(1)
        # choose random
        r = np.random.choice(3)
        mode = row.most_common(r + 1)[0][0]
        print 'mode', mode
      labels[i] = mode
    except:
      labels[i] = 39 # unclear or missing image name in annotations data
      missing_images.append(subfolder + '/' + img)
    print 'label', i, labels[i]

  print 'labels'
  labels = labels.astype(int)

  print labels
  return labels

 

  """
  for i in range(labels.shape[0]):
    row = Counter(data[img_names[i]])
    #print row.most_common
    
    mode = row.most_common(1)[0][0]
    print labels[i]
    print data[img_names[i]]
    if row.most_common(1)[0][1] == 1:
      print row.most_common(1)
      # choose random
      r = np.random.choice(3)
      print 'r', r
      mode = row.most_common(r + 1)[0][0]

    print 'mode', mode

    try:
      labels[i] = mode
    except:
      labels[i] = 39 # unclear
  """    

if __name__ == '__main__':

  with open('Topics.json') as data:
    data = json.load(data)
 
  missing_images = []
  for i in range(11):
    if i != 0:
      continue
    subfolder = str(i)
    input_file_path = '/home/micha/project/' + subfolder + '/'
    labels = get_labels(input_file_path, data, missing_images)
    pickle.dump(labels, open('labels/' + subfolder + '_labels.pkl', 'wb'))
 
  print 'missing images'
  print missing_images
  print len(missing_images)
  #img_names = [str(d) for d in data.keys()]
  #print img_names[:10]
  #labels = get_labels(input_file_path)

  #pickle.dump(labels, open('labels.pkl', 'wb'))



