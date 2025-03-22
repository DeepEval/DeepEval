import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def repeated_block(input_tensor, filters):
        # Batch normalization and ReLU activation, followed by a 3x3 convolutional layer
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.activations.relu(batch_norm)
        conv = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)
        return max_pooling

    # Pathway 1
    pathway1_input = Input(shape=(28, 28, 1), name='Pathway1')
    pathway1_output = repeated_block(input_tensor=pathway1_input, filters=64)

    # Pathway 2
    pathway2_input = Input(shape=(28, 28, 1), name='Pathway2')
    pathway2_output = repeated_block(input_tensor=pathway2_input, filters=64)

    # Concatenate the outputs of both pathways
    concatenated_output = Concatenate(axis=1)([pathway1_output, pathway2_output])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model architecture
    model = keras.Model(inputs=[pathway1_input, pathway2_input], outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])