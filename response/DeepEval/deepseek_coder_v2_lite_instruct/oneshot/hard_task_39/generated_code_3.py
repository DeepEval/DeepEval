import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dense

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Max pooling layers with different sizes
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    # Flatten and concatenate the outputs
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer after Block 1
    dense1 = Dense(units=128, activation='relu')(concatenated)
    
    # Reshape the output to 4D tensor suitable for Block 2
    reshape_layer = Reshape((4, 4, 1))(dense1)
    
    # Block 2
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)
    
    # Concatenate the outputs from all branches
    concat_block2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated output
    flatten_block2 = Flatten()(concat_block2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_block2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model