import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, LayerNormalization

def dl_model():
  # Input layer
  input_layer = Input(shape=(32, 32, 3))

  # Depthwise separable convolutional layer
  conv = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu')(input_layer)

  # Layer normalization
  layer_norm = LayerNormalization()(conv)

  # Max pooling layer
  max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(layer_norm)

  # Fully connected layers
  dense1 = Dense(units=32, activation='relu')(max_pooling)
  dense2 = Dense(units=32, activation='relu')(dense1)

  # Add original input to processed features
  concat = Concatenate()([input_layer, dense2])

  # Output layer
  output_layer = Dense(units=10, activation='softmax')(concat)

  # Model
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model