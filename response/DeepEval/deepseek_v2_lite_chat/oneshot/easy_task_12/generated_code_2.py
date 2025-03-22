import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(conv1)
        batch_norm1 = BatchNormalization()(sep_conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)

        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
        batch_norm2 = BatchNormalization()(sep_conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)

        return pool2

    # Branch path
    def branch_path(input_tensor):
        conv_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv_branch

    # Combine outputs from main path and branch path
    main_output = main_path(input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    combined_output = Concatenate()([main_output, branch_output])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()