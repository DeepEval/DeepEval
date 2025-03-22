import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv_main = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    layer_norm_main = BatchNormalization()(conv_main)
    
    conv_main_pointwise1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm_main)
    conv_main_pointwise2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main_pointwise1)
    
    # Branch path
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    # Combine paths
    concat = Add()([conv_main_pointwise2, conv_branch])
    
    # Fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model