import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  input_layer = layers.Input(shape=(32, 32, 3))

  # Convolutional layer with initial feature extraction
  conv_layer = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)

  # Batch normalization
  bn_layer = layers.BatchNormalization()(conv_layer)

  # Global average pooling for feature compression
  gap_layer = layers.GlobalAveragePooling2D()(bn_layer)

  # Fully connected layers for further dimensionality adjustment
  fc1_layer = layers.Dense(32, activation='relu')(gap_layer)
  fc2_layer = layers.Dense(32, activation='relu')(fc1_layer)

  # Reshape and concatenate with initial features
  reshaped_output = layers.Reshape((32, 32, 32))(fc2_layer)
  weighted_features = layers.Multiply()([reshaped_output, conv_layer])
  concatenated_features = layers.Concatenate()([input_layer, weighted_features])

  # Dimensionality reduction and downsampling
  conv_downsample = layers.Conv2D(16, (1, 1), strides=(2, 2))(concatenated_features)
  avg_pool_downsample = layers.AveragePooling2D(pool_size=(2, 2))(conv_downsample)

  # Final fully connected layer for classification
  output_layer = layers.Dense(10, activation='softmax')(avg_pool_downsample)

  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  return model