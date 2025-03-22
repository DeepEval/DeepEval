import keras
from keras.layers import Input, Conv2D, Lambda, Reshape, Permute, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Block 1
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    # Main path
    block1_output = block1(input_layer)
    block2_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block3_output = Reshape((-1, 3, int(block1_output.shape[-1] / 3)))(block2_output)
    block3_output = Permute((3, 2, 1))(block3_output)
    block3_output = Reshape((block2_output.shape[1], block2_output.shape[2], -1))(block3_output)
    block3_output = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3_output)
    
    # Branch
    branch_output = input_layer
    
    # Combine the main path and branch
    combined_output = Add()([block3_output, branch_output])
    
    # Flatten and Dense layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model