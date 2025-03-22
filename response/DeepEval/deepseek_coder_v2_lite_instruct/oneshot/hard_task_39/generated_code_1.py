import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    concat1 = Concatenate()([flatten1, flatten2, flatten3])
    
    fc1 = Dense(units=128, activation='relu')(concat1)
    reshape_layer = Reshape((1, 1, 3))(fc1)  # Convert to 4D tensor
    
    # Block 2
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)
    
    concat2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    flatten2 = Flatten()(concat2)
    fc2 = Dense(units=64, activation='relu')(flatten2)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model