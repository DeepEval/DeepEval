import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: four parallel branches
    branch1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs from the parallel branches
    merged = Concatenate(axis=-1)([branch1_1, branch1_2, branch1_3, branch1_4])
    
    # Second block: global average pooling and two fully connected layers
    pooled_features = GlobalAveragePooling2D()(merged)
    dense1 = Dense(units=128, activation='relu')(pooled_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape weights to match the input shape of the second block
    weights_shape = (1, 1, 1, 64)  # Adjust these values based on the actual number of filters in the second block
    reshaped_weights = keras.layers.Reshape(target_shape=weights_shape)(dense2)
    
    # Element-wise multiplication with the input feature map
    output = keras.layers.multiply([reshaped_weights, pooled_features])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model