import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten and concatenate the outputs from the pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(concatenated)
    
    # Reshape to prepare for the second block
    reshape_layer = Reshape((1, 1, 128))(dense1)
    
    # Second block: Four parallel paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    path1 = Dropout(0.5)(path1)

    path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    path2 = Dropout(0.5)(path2)

    path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path3)
    path3 = Dropout(0.5)(path3)

    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshape_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)
    path4 = Dropout(0.5)(path4)

    # Concatenate all paths
    concatenated_paths = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flatten_paths = Flatten()(concatenated_paths)

    # Fully connected layers for classification
    dense2 = Dense(units=64, activation='relu')(flatten_paths)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model