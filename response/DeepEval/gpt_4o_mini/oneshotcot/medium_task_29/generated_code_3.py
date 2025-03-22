import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():     
    # Input layer for CIFAR-10 images (32x32 pixels with 3 color channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # Max pooling layers with varying window sizes
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling output
    flatten1 = Flatten()(max_pooling1)
    flatten2 = Flatten()(max_pooling2)
    flatten3 = Flatten()(max_pooling3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model