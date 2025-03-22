import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        group_size = 3
        split = Lambda(lambda x: K.tf.split(x, group_size, axis=-1))(input_tensor)
        conv = Conv2D(filters=int(input_tensor.shape[-1] / group_size), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        output_tensor = Concatenate()([conv] * group_size)
        return output_tensor
        
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        shape = K.int_shape(input_tensor)
        reshaped = Lambda(lambda x: K.reshape(x, (-1, shape[1], shape[2], shape[3] // 3, 3)))(input_tensor)
        swapped = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 2, 4, 3)))(reshaped)
        reshaped_back = Lambda(lambda x: K.reshape(x, (-1, shape[1], shape[2], shape[3])))(swapped)
        return reshaped_back
        
    block2_output = block2(block1_output)
    
    # Block 3
    block3_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise=True, use_bias=False)(block2_output)
    
    # Average pooling branch
    branch_output = AveragePooling2D(pool_size=(8, 8), strides=8, padding='valid')(input_layer)
    
    # Concatenate main path and branch path
    output_tensor = Concatenate()([block3_output, branch_output])
    
    # Apply batch normalization and flatten
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model