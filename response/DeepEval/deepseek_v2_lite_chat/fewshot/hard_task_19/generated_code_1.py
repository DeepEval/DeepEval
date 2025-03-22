import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv3)

        return pool1

    branch_output = main_path(input_tensor=input_layer)

    # Branch path
    def branch_path(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(input_tensor)
        flat1 = Flatten()(avg_pool)
        dense1 = Dense(units=128, activation='relu')(flat1)
        dense2 = Dense(units=64, activation='relu')(dense1)
        weights = Dense(units=3, activation=None)(dense2)  # Generate channel weights
        reshaped_weights = Reshape((-1, 1))(weights)  # Reshape weights for element-wise multiplication
        channel_weights = keras.layers.Activation('sigmoid')(reshaped_weights)  # Apply sigmoid activation
        branch_output = channel_weights * input_tensor  # Multiply input with channel weights

        return branch_output

    branch_output = branch_path(input_tensor=branch_output)

    # Add outputs from both paths
    combined_output = keras.layers.Add()([branch_output, conv2])

    # Pass through additional fully connected layers
    dense1 = Dense(units=256, activation='relu')(combined_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()