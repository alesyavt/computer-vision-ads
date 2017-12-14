import os
import sys
import csv
import pickle
import numpy as np
import scipy.io
import scipy.spatial.distance
import tensorflow as tf
# This should be a path to the "slim" directory in tensorflow models which can be cloned from here: https://github.com/tensorflow/models
sys.path.append('../workspace/models/research/slim')
from nets import vgg as vgg
from preprocessing import vgg_preprocessing as preproc





def unpickle(filename):
  with open(filename, 'rb') as fo:
    datadict = pickle.load(fo)
  return datadict

def extract_features(input_file_path):
  image_names = os.listdir(input_file_path)
  print 'len image list', len(image_names)

  image_list = list()

  num_images = len(image_names)
  for i in range(num_images):
    img = image_names[i]
    image_list.append(os.path.join(input_file_path, img))
  print image_list[:10]
  images = image_list
  
  slim = tf.contrib.slim

  # Get the image size that vgg_19 accepts
  image_size = vgg.vgg_19.default_image_size
  preprocessed_images = list()
  out_features_list = []


  total_count = 0
  batch = 0
  while total_count < len(images):
    batch += 1
    print 'batch number', batch
  
    preprocessed_images = list()

    with tf.Graph().as_default():
      # This allows for default parameters
      with slim.arg_scope(vgg.vgg_arg_scope()):

        for c in range(10): 
          if total_count >= len(images):
            break  ##
          print total_count
          print images[total_count]
          image = tf.read_file(image_list[total_count])
          decoded_image = tf.image.decode_jpeg(image, channels=3)
          preprocessed_images.append(preproc.preprocess_image(decoded_image, image_size, image_size, is_training=True))
          total_count += 1

        stacked_images = tf.stack(preprocessed_images)
        print 'stacked images', stacked_images
        _, end_points = vgg.vgg_19(stacked_images, is_training=False)

	with tf.Session() as sess: 
	  print 'inside tf sess'
    sess.run(tf.global_variables_initializer())
	  saver = tf.train.Saver()
	  out_features = sess.run(stacked_images)

    out_features_list.extend(out_features)

    print 'accumulated features array'
    print np.array(out_features_list).shape
  return np.array(out_features_list)



if __name__ == '__main__':

  # just work with 0/ image subfolder
  for i in range(11):
    if i != 0:
      continue
    subfolder = str(i)
    input_file_path = '/home/micha/project/' + subfolder + '/'

    # Note: here features are actually the resized images
    features = extract_features(input_file_path)
    
    image_names = os.listdir(input_file_path)
    print 'len image list', len(image_names)

    image_list = list()

    num_images = len(image_names)
    for i in range(num_images):
      img = image_names[i]
      image_list.append(os.path.join(input_file_path, img))
    #print image_list[:10]
    images = image_list

    
    print 'writing to out path'
    # Change filename in out_path variable
    
    # Indexing the images for predictions
    # save image names order
    pickle.dump(np.array(image_names), open('image-index-to-names.pkl', 'wb'))
    f = open('image-index-to-names.txt', 'wb')
    for i, n in enumerate(image_names):
      f.write(str(i) + '  ' + n + '\n')
    f.close()

    out_path = 'preprocessed-vgg/' + subfolder + '_stacked_images_train.pkl'
    pickle.dump(features, open(out_path, 'wb'))







