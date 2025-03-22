import keras
from keras.layers import Input, Conv2D, ReLU, Add, concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def block(input_tensor):
        # First block in main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Second block in main path
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Concatenate the outputs of the two blocks along the channel dimension
        concat = concatenate([conv1, conv2])
        return concat

    main_path_output = block(input_tensor=input_layer)

    # Branch path
    def branch_block(input_tensor):
        # Block in branch path
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Concatenate the outputs of the block with the main path's output along the channel dimension
        concat = concatenate([conv3, main_path_output])
        return concat

    branch_output = branch_block(input_tensor=input_layer)

    # Fusion layer
    fusion_output = Add()([main_path_output, branch_output])

    # Flatten and fully connected layer
    flatten = Flatten()(fusion_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])