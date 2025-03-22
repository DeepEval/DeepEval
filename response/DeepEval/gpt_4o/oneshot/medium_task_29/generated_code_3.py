import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Applying max pooling with different window sizes
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flattening the pooled outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenating the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model