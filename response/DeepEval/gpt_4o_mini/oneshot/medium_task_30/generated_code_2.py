import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Average pooling layers with different pooling windows and strides
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of the pooling layers
    flatten_pool_1 = Flatten()(avg_pool_1)
    flatten_pool_2 = Flatten()(avg_pool_2)
    flatten_pool_3 = Flatten()(avg_pool_3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten_pool_1, flatten_pool_2, flatten_pool_3])

    # Further flattening is not necessary after concatenation, just pass through fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model