import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        tensor_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_split[0])
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_split[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_split[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        shape = layers.InputShape(input_tensor.shape[1:-1] + [int(input_tensor.shape[-1] / 3)] * 3).shape
        reshaped_tensor = Reshape(shape)(input_tensor)
        permuted_tensor = Permute((2, 3, 1))(reshaped_tensor)
        return permuted_tensor
    
    block2_output = block2(block1_output)
    
    # Block 3
    def block3(input_tensor):
        depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', depthwise=True)(input_tensor)
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(depthwise_conv)
        return output_tensor
    
    block3_output = block3(block2_output)
    
    # Block 4 (Repeat Block 1)
    block4_output = block1(block3_output)
    
    # Branch Path
    average_pooling = MaxPooling2D(pool_size=(8, 8))(input_layer)
    
    # Concatenate the outputs from both paths
    combined_output = Concatenate()([block4_output, average_pooling])
    
    # Batch Normalization
    bath_norm = BatchNormalization()(combined_output)
    
    # Flatten
    flatten_layer = Flatten()(bath_norm)
    
    # Dense Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model