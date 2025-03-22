import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.initializers import he_normal

def depthwise_conv2d(input_tensor, kernel_size, strides, padding):
  return DepthwiseConv2D(kernel_size, strides=strides, padding=padding)(input_tensor)

def pointwise_conv2d(input_tensor, filters, strides, padding):
  return Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding=padding)(input_tensor)

def squeeze_excite(input_tensor, ratio):
  filters = int(input_tensor.shape[-1] * ratio)
  excitation = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
  excitation = Activation('relu')(excitation)
  excitation = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(excitation)
  excitation = Activation('sigmoid')(excitation)
  return keras.layers.multiply([input_tensor, excitation])

def block(input_tensor, filters, strides):
  main_path = input_tensor

  # Main path: 
  main_path = pointwise_conv2d(main_path, filters=filters, strides=strides, padding='same')
  main_path = Activation('relu')(main_path)
  main_path = depthwise_conv2d(main_path, kernel_size=(3, 3), strides=(1, 1), padding='same')
  main_path = Activation('relu')(main_path)
  main_path = pointwise_conv2d(main_path, filters=filters, strides=(1, 1), padding='same')
  main_path = BatchNormalization()(main_path)

  # Branch path:
  branch_path = pointwise_conv2d(input_tensor, filters=filters, strides=strides, padding='same')
  branch_path = Activation('relu')(branch_path)
  branch_path = depthwise_conv2d(branch_path, kernel_size=(3, 3), strides=(1, 1), padding='same')
  branch_path = Activation('relu')(branch_path)
  branch_path = pointwise_conv2d(branch_path, filters=filters, strides=(1, 1), padding='same')
  branch_path = BatchNormalization()(branch_path)

  # Fuse paths:
  fused_path = keras.layers.add([main_path, branch_path])
  fused_path = Activation('relu')(fused_path)
  fused_path = squeeze_excite(fused_path, ratio=0.25)

  return fused_path

def dl_model():
  input_layer = Input(shape=(28, 28, 1))
  conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
  max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

  block_output = block(input_tensor=max_pooling, filters=64, strides=1)
  block_output = block(input_tensor=block_output, filters=64, strides=1)
  block_output = block(input_tensor=block_output, filters=64, strides=1)

  flatten_layer = Flatten()(block_output)
  dense1 = Dense(units=128, activation='relu')(flatten_layer)
  dense2 = Dense(units=64, activation='relu')(dense1)
  output_layer = Dense(units=10, activation='softmax')(dense2)

  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model

model = dl_model()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

score = model.evaluate(x_test, y_test)

print('Test accuracy:', score[1])