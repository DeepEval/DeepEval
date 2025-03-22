import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Splitting and processing input
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv_1 = [Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in split_1]
    concatenated_1 = Concatenate(axis=-1)(conv_1)
    
    # Block 2: Channel shuffling and reshaping
    reshaped_1 = Lambda(lambda x: tf.transpose(x[:, :, :, :int(x.get_shape()[3] / 3)]))(concatenated_1)
    reshaped_1 = Permute((2, 3, 1))(reshaped_1)  # Swap third and first dimensions
    reshaped_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_1)
    
    # Block 3: Depthwise separable convolution
    depthwise_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                             separable=True)(reshaped_1)
    
    # Branch connecting directly to the input
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine outputs from main path and branch
    combined = Concatenate(axis=-1)([depthwise_conv, branch_input])
    
    # Final dense layers for classification
    dense1 = Dense(units=128, activation='relu')(combined)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model