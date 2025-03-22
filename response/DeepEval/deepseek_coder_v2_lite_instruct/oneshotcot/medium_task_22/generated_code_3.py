import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch with 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Second branch with 1x1 convolution followed by two 3x3 convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Third branch with max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Concatenate the outputs of the branches
    merged = Concatenate()([branch1, branch2, branch3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(merged)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model