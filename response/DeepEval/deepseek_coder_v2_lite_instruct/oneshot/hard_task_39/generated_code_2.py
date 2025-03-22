import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    def block1(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        concatenated = Concatenate()([flatten1, flatten2, flatten3])
        return concatenated
    
    block1_output = block1(input_tensor=input_layer)
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshape_output = Reshape((1, 1, 128))(dense1)
    
    # Second block
    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        concatenated = Concatenate()([path1, path2, path3, path4])
        return concatenated
    
    block2_output = block2(input_tensor=reshape_output)
    flatten_output = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model