import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Dense, Dropout, Reshape

def dl_model():

  input_layer = Input(shape=(28,28,1))

  # Average pooling layer with 5x5 window and 3x3 stride
  avg_pool_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)

  # 1x1 convolutional layer
  conv_1 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(avg_pool_1)

  # Flatten the feature maps
  flatten = Flatten()(conv_1)

  # Fully connected layer 1
  dense_1 = Dense(units=64, activation='relu')(flatten)

  # Dropout layer
  dropout = Dropout(rate=0.25)(dense_1)

  # Fully connected layer 2
  dense_2 = Dense(units=32, activation='relu')(dropout)

  # Output layer for multi-class classification
  output_layer = Dense(units=10, activation='softmax')(dense_2)

  # Create the model
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model