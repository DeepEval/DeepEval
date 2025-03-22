import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 color channels

    # Average pooling layers with specified pooling windows and strides
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs of the pooling layers
    flat_1x1 = Flatten()(avg_pool_1x1)
    flat_2x2 = Flatten()(avg_pool_2x2)
    flat_4x4 = Flatten()(avg_pool_4x4)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Process through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model