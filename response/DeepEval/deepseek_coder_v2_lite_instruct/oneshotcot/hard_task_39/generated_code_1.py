import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Process through three max pooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    # Flatten and concatenate the results from each pooling layer
    concat_layer_1 = Concatenate()([Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])
    
    # Fully connected layer to transform the output of Block 1
    dense1 = Dense(units=128, activation='relu')(concat_layer_1)
    reshape_layer = Reshape((1, 1, 128))(dense1)  # Convert to 4-dimensional tensor
    
    # Block 2: Multiple branches for processing
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)
    
    # Concatenate the outputs from all branches
    concat_layer_2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated tensor
    flatten_layer = Flatten()(concat_layer_2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model