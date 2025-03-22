import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Reshape, Permute, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
      inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
      conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
      conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
      conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
      output_tensor = Concatenate()([conv1, conv2, conv3])
      return output_tensor

    
    block1_output = block1(input_layer)

    def block2(input_tensor):
      shape = tf.shape(input_tensor)
      reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
      permuted = Permute(axes=[0, 1, 3, 2])(reshaped)
      output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
      return output_tensor

    block2_output = block2(block1_output)

    def block3(input_tensor):
      depthwise_conv = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(input_tensor)
      branch_output = input_layer
      output_tensor = Add()([depthwise_conv, branch_output])
      return output_tensor

    block3_output = block3(block2_output)


    block1_output_again = block1(block3_output)
    flatten_layer = Flatten()(block1_output_again)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model