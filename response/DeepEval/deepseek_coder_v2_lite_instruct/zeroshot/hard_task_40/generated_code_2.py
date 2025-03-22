import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, Concatenate

def dl_model():
    # First block
    inputs = Input(shape=(28, 28, 1))
    
    # Pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(inputs)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(inputs)
    
    # Flatten and concatenate outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape
    dense1 = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((1, 1, 384))(dense1)  # Assuming the output of dense1 is 384
    
    # Second block
    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path1_dropout = Dropout(0.5)(path1)
    
    # Path 2: 1x1 followed by two 3x3 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2_dropout = Dropout(0.5)(path2)
    
    # Path 3: 1x1 followed by a single 3x3 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3_dropout = Dropout(0.5)(path3)
    
    # Path 4: 1x1 convolution followed by average pooling
    path4 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path4 = AveragePooling2D(pool_size=(2, 2), strides=2)(path4)
    path4_dropout = Dropout(0.5)(path4)
    
    # Concatenate outputs from all paths
    concatenated_paths = Concatenate(axis=-1)([path1_dropout, path2_dropout, path3_dropout, path4_dropout])
    
    # Fully connected layers
    dense2 = Dense(128, activation='relu')(concatenated_paths)
    outputs = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()