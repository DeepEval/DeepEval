import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    def block(input_tensor):
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        return pool2

    block_output1 = block(pool1)
    block_output2 = block(pool1)

    # Concatenate the outputs of the two blocks
    concat = Concatenate()([block_output1, block_output2])

    # Add a fully connected layer after the concatenation
    dense = Dense(units=128, activation='relu')(concat)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])