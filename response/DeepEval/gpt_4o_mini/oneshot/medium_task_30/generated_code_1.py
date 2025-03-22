import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Average Pooling Layers
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of the pooling layers
    flatten_1 = Flatten()(avg_pool_1)
    flatten_2 = Flatten()(avg_pool_2)
    flatten_3 = Flatten()(avg_pool_3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten_1, flatten_2, flatten_3])

    # Further flatten (although already flattened, just for clarity)
    flatten_final = Flatten()(concatenated)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_final)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()  # To show the model summary