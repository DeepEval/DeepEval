import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dense, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    # Flatten and concatenate results from Block 1
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concat_layer1 = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape to prepare for Block 2
    dense1 = Dense(units=128, activation='relu')(concat_layer1)
    reshape_layer = Reshape((1, 1, 128))(dense1)  # Reshape to 4D tensor
    
    # Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block2_output = block2(reshape_layer)
    flatten_layer = Flatten()(block2_output)
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model