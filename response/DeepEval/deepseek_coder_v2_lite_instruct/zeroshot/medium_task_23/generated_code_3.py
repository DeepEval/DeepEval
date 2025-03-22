import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu')(path2)
    
    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3 = Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu')(path3)
    
    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D(pool_size=(8, 8))(input_layer)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    
    # Concatenate the outputs of the paths
    concatenated = concatenate([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()