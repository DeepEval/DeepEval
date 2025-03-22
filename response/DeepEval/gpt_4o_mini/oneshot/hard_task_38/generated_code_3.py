import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def block(input_tensor):
    # Apply Batch Normalization and ReLU activation
    batch_norm = BatchNormalization()(input_tensor)
    relu = ReLU()(batch_norm)
    
    # Convolutional layer that preserves spatial dimensions
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
    
    # Concatenate the input tensor with the new features
    output_tensor = Concatenate()([input_tensor, conv])
    
    return output_tensor

def dl_model():
    # Input layer for MNIST images
    input_layer = Input(shape=(28, 28, 1))

    # First processing pathway
    path1 = input_layer
    for _ in range(3):
        path1 = block(path1)

    # Second processing pathway
    path2 = input_layer
    for _ in range(3):
        path2 = block(path2)

    # Concatenate both pathways
    merged_output = Concatenate()([path1, path2])
    
    # Flatten the merged output
    flatten_layer = Flatten()(merged_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model