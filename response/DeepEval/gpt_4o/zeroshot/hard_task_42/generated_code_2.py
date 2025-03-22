from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input Layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three Parallel Paths with Max Pooling
    # Path 1: 1x1 Pooling
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: 2x2 Pooling
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: 4x4 Pooling
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate outputs from all paths
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer and reshape to 4D tensor
    fc = Dense(256, activation='relu')(block1_output)
    reshape = Reshape((4, 4, 16))(fc)  # Reshape to 4D, adjust dimensions based on your architecture
    
    # Block 2: Four Parallel Paths with Convolution and Pooling
    # Path 1: 1x1 Convolution
    path1_b2 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshape)
    
    # Path 2: 1x1 -> 1x7 -> 7x1 Convolution
    path2_b2 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshape)
    path2_b2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path2_b2)
    path2_b2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path2_b2)
    
    # Path 3: 1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7 Convolution
    path3_b2 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshape)
    path3_b2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path3_b2)
    
    # Path 4: Average Pooling -> 1x1 Convolution
    path4_b2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape)
    path4_b2 = Conv2D(32, (1, 1), activation='relu', padding='same')(path4_b2)

    # Concatenate outputs from all paths in Block 2
    block2_output = Concatenate(axis=-1)([path1_b2, path2_b2, path3_b2, path4_b2])

    # Fully connected layers for classification
    fc1 = Flatten()(block2_output)
    fc1 = Dense(128, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Construct the Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model