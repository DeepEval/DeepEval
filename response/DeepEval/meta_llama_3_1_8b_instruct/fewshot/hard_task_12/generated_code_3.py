import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    concat_layer = Concatenate()([conv1_2, conv2])
    
    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge the outputs from both paths
    adding_layer = Add()([concat_layer, branch_path])
    
    # Classification path
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model