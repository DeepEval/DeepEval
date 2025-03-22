import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Concatenate, Flatten, multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Convolution, Global Average Pooling, and Fully Connected Layers
    def branch1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        avg_pool = GlobalAveragePooling2D()(conv1)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)

        return dense2

    branch1_output = branch1(input_tensor=input_layer)

    # Branch 2: Global Average Pooling, Fully Connected Layers, and the Block
    def branch2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        block_output = branch1(input_tensor=input_tensor)

        return Concatenate()([block_output, dense2])

    branch2_output = branch2(input_tensor=input_layer)

    # Final Dense Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(branch2_output)

    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Print Model Summary
model.summary()