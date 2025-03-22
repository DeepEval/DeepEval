import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Convolution and Average Pooling
    def conv_block(x):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    
    path1_output = conv_block(input_layer)
    
    # Add more blocks for deeper feature extraction if needed
    for _ in range(5):  # Add 5 more blocks
        path1_output = conv_block(path1_output)
    
    # Flatten and fully connected layers
    x = Flatten()(path1_output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Path 2: Single Convolution
    path2_output = Conv2D(64, (3, 3), padding='same')(input_layer)
    path2_output = BatchNormalization()(path2_output)
    path2_output = keras.activations.relu(path2_output)
    
    # Combine Path 1 and Path 2 outputs
    combined_output = Add()([path1_output, path2_output])
    
    # Batch normalization and fully connected layers
    x = BatchNormalization()(combined_output)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=x)
    
    return model