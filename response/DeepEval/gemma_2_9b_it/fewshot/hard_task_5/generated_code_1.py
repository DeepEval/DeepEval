import keras
from keras.layers import Input, Conv2D, Lambda, Reshape, Permute, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
      inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
      conv1 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
      conv2 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
      conv3 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
      output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
      return output_tensor

    block1_output = block1(input_layer)

    def block2(input_tensor):
      shape = Lambda(lambda x: tf.shape(x))(input_tensor)
      reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3]//3))(input_tensor)
      permuted = Permute((1, 2, 4, 3))(reshaped)
      reshaped = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
      return reshaped

    block2_output = block2(block1_output)

    def block3(input_tensor):
      depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      branch = input_tensor
      output_tensor = Add()([depthwise_conv, branch])
      return output_tensor

    block3_output = block3(block2_output)

    block4_output = block1(block3_output)

    flatten = Flatten()(block4_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model