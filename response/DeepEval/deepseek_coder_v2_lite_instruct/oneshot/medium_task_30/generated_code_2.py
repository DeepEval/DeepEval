import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First average pooling layer
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    
    # Second average pooling layer
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    
    # Third average pooling layer
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(conv3)
    
    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concat_layer = Concatenate()([flatten1, flatten2, flatten3])
    
    # Flatten the concatenated output
    flatten_concat = Flatten()(concat_layer)
    
    # Two fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model