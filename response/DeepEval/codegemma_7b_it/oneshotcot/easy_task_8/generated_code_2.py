import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
  # Create the input layer
  input_layer = Input(shape=(28, 28, 1))

  # Add a depthwise separable convolutional layer
  conv_dw = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(input_layer)

  # Add a 1x1 convolutional layer for feature extraction
  conv_pw = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', use_bias=True)(conv_dw)

  # Add a dropout layer
  dropout = Dropout(rate=0.25)(conv_pw)

  # Add a max pooling layer
  max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout)

  # Flatten the output
  flatten_layer = Flatten()(max_pooling)

  # Add a fully connected layer
  dense = Dense(units=10, activation='softmax')(flatten_layer)

  # Create the model
  model = keras.Model(inputs=input_layer, outputs=dense)

  return model