import keras
from keras.layers import Input, Lambda, Conv2D, Reshape, Permute, AveragePooling2D, concatenate, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = concatenate([conv1, conv2, conv3], axis=-1)
        return output_tensor

    def block2(input_tensor):
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, input_tensor.shape[-1]//3))(input_tensor)
        permuted = Permute(dims=(0, 1, 3, 2))(reshaped)
        return Reshape(target_shape=(shape[1], shape[2], input_tensor.shape[-1]))(permuted)

    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(block2_output)
    
    branch_output = input_layer

    adding_layer = concatenate([block3_output, branch_output], axis=-1)
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model