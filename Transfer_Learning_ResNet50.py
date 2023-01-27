# from matplotlib.backend_bases import ResizeEvent

## Import library
import numpy as np
import tensorflow as tf
import os, re, time, json
import matplotlib.pyplot as plt
#import tensorflow_datasets as tfds
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
from tensorflow.keras.applications.resnet50 import ResNet50


# Parameters
BATCH_SIZE = 32 
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Loading and Preprocessing Data
(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

# utility to display training and validation curves

def plot_metrics(metric_name, title, ylim=5):
  plt.title(title)
  plt.ylim(0,ylim)
  plt.plot(history.history[metric_name],color='blue',label=metric_name)
  plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

def display_images(digits, predictions, labels, title):

  n = 10

  indexes = np.random.choice(len(predictions), size=n)
  n_digits = digits[indexes]
  n_predictions = predictions[indexes]
  n_predictions = n_predictions.reshape((n,))
  n_labels = labels[indexes]
 
  fig = plt.figure(figsize=(20, 4))
  plt.title(title)
  plt.yticks([])
  plt.xticks([])

  for i in range(10):
    ax = fig.add_subplot(1, 10, i+1)
    class_index = n_predictions[i]
    
    plt.xlabel(classes[class_index])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(n_digits[i])

# Preprocess Dataset 
def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

# Define the Network

def feature_extractor(inputs):
  feature_extractor = ResNet50(input_shape = (224,224,3),
                               include_top = False,
                               weights = 'imagenet')(inputs)
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
  resize = tf.keras.layers.UpSampling2D(size = (7,7))(inputs)
  resnet_feature_extractor = feature_extractor(resize)
  classification_output = classifier(resnet_feature_extractor)
  return classification_output


def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  outputs = final_model(inputs)
  model = tf.keras.Model(inputs = inputs, outputs = outputs)
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
  return model


model = define_compile_model()

# Train the model

epoch = 5
histoy = model.fit(train_X, training_labels, epochs = epoch, validation_data = (valid_X, validation_labels), batch_size=64)

# Evaluate the Model
loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

# Plot Loss and Accuracy Curves
plot_metrics("loss", "Loss")
plot_metrics("accuracy", "Accuracy")

# Visualize predictions
probabilities = model.predict(valid_X, batch_size=64)
probabilities = np.argmax(probabilities, axis = 1)

display_images(validation_images, probabilities, validation_labels, "Bad predictions indicated in red.")