import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolutions
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2: 1x1 Convolutions + 3x3 Convolutions
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    
    # Concatenate Branches
    concat_output = Concatenate()([branch1, branch2, branch3])
    
    # Batch Normalization and Flatten
    concat_output = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(concat_output)
    
    # Fully Connected Layers
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example Usage
model = dl_model()