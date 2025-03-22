import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate
from keras.models import Model

def dl_model():
  # Define the input layer
  input_layer = Input(shape=(32, 32, 3))

  # Extract initial features
  conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
  bn = BatchNormalization()(conv)
  relu = Activation('relu')(bn)

  # Compress feature maps
  gap = GlobalAveragePooling2D()(relu)
  fc1 = Dense(units=64, activation='relu')(gap)
  fc2 = Dense(units=64, activation='relu')(fc1)

  # Reshape and generate weighted feature maps
  fc2_reshape = Reshape((1, 1, 64))(fc2)
  weighted_feature_maps = Multiply()([relu, fc2_reshape])

  # Concatenate weighted feature maps with input
  concat = Concatenate()([input_layer, weighted_feature_maps])

  # Reduce dimensionality and downsample
  conv_downsample = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
  avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_downsample)

  # Classification layer
  flatten = Flatten()(avg_pool)
  output_layer = Dense(units=10, activation='softmax')(flatten)

  # Create the model
  model = Model(inputs=input_layer, outputs=output_layer)

  return model