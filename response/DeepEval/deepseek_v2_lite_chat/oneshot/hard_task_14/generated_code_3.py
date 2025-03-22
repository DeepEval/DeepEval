import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Convolution layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # Global average pooling
        avg_pool1 = GlobalAveragePooling2D()(pool1)

        # Fully connected layer 1
        dense1 = Dense(units=128, activation='relu')(avg_pool1)
        # Fully connected layer 2
        dense2 = Dense(units=64, activation='relu')(dense1)

        return dense2

    # Branch path
    def branch_path(input_tensor):
        # Convolution layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Fully connected layer 1
        dense1 = Dense(units=128, activation='relu')(conv1)

        return dense1

    # Combine paths
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    combined_output = Concatenate()([main_output, branch_output])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(combined_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()