import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction path 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
    
    # Concatenate the outputs from both paths
    output_tensor = Concatenate()([conv1, conv4])
    
    # Apply 1x1 convolution to align the output dimensions
    output_tensor = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(output_tensor)
    
    # Define the branch that connects directly to the input
    branch = input_layer
    
    # Merge the outputs through addition
    merged_tensor = Add()([output_tensor, branch])
    
    # Batch normalization
    bath_norm = BatchNormalization()(merged_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model