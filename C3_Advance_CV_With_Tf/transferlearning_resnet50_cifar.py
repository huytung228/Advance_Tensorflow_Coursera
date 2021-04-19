# This module using transfer learning Resnet50 for classification CIFAR10 dataset
# Import needed modules
import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds 

# Load CIFAR dataset
def load_CIFAR_dataset():
    # Load from keras
    (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
    return (training_images, training_labels) , (validation_images, validation_labels)

# perform normalization on images in training and validation set
def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

# Define network
'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
include_top = False -> dont use last layer of Resnet, last layer of original resnet for classification 1000 classes
weight -> using pretrain imagenet weight 
'''
def feature_extractor(inputs):

  feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
  return feature_extractor


'''
Defines final dense layers and subsequent softmax layer for classification.
'''
def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''
def final_model(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''
def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  
  classification_output = final_model(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification_output)
 
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
  return model

if __name__ == "__main__":
    model = define_compile_model()
    (training_images, training_labels) , (validation_images, validation_labels) = load_CIFAR_dataset()
    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)
    history = model.fit(train_X, training_labels, epochs=4, validation_data = (valid_X, validation_labels), batch_size=64)
    loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

