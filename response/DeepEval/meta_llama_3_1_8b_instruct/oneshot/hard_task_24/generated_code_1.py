import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: Local Features
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    
    # Branch 2 & 3: Downsampling, then upsampling
    max_pool_branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch2)
    upsample_branch2 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv_branch2)

    max_pool_branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    conv_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch3)
    upsample_branch3 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv_branch3)

    # Concatenate the outputs of all branches
    output_tensor = Concatenate()([conv_branch1, upsample_branch2, upsample_branch3])

    # Apply another 1x1 convolution
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    # Flatten the output
    flatten_layer = Flatten()(final_conv)
    
    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model