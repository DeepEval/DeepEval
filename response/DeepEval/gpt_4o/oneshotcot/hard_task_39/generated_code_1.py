import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate

def dl_model():
    # Input Layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Multi-scale Max Pooling
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pool4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten each pooling output
    flat1 = Flatten()(max_pool1)
    flat2 = Flatten()(max_pool2)
    flat4 = Flatten()(max_pool4)
    
    # Concatenate flattened results
    block1_output = Concatenate()([flat1, flat2, flat4])
    
    # Fully connected layer and reshape to 4D for Block 2
    dense_block1 = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape(target_shape=(8, 8, 2))(dense_block1)  # Example reshape, adjust based on needs
    
    # Block 2: Convolutional and Pooling branches
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    max_pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(reshape_layer)
    
    # Concatenate all branches
    block2_output = Concatenate()([conv1x1, conv3x3, conv5x5, max_pool3x3])
    
    # Flatten and Fully connected output layer
    flat_final = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flat_final)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model