import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 Convolution
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1 - Local features through 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    
    # Branch 2 - Downsample, 3x3 Conv, Upsample
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_up = UpSampling2D(size=(2, 2))(branch2_conv)
    
    # Branch 3 - Downsample, 3x3 Conv, Upsample
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_up = UpSampling2D(size=(2, 2))(branch3_conv)
    
    # Concatenate all branches
    concat_branches = Concatenate()([branch1, branch2_up, branch3_up])
    
    # 1x1 Convolution after concatenation
    post_concat_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)
    
    # Fully connected layers
    flatten_layer = Flatten()(post_concat_conv)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model