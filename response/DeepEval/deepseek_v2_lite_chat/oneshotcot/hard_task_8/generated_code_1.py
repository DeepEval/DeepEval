import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Main path
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1_1)
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv1_2)

        # Branch path
        conv2_1 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv2_1)

        # Concatenate features from both paths
        concat_tensor = Concatenate(axis=-1)([conv1_3, conv2_2])
        
        return concat_tensor

    # Block 2
    def block2(input_tensor):
        # Get the shape of features from Block 1
        shape = keras.backend.int_shape(input_tensor)
        
        # Reshape and shuffle features
        reshaped_tensor = keras.backend.reshape(input_tensor, shape=(shape[0], shape[1], shape[3], shape[2]))
        shuffled_tensor = Permute((2, 3, 1))(reshaped_tensor)
        
        # Flatten and feed into a fully connected layer
        flatten_tensor = Flatten()(shuffled_tensor)
        dense1 = Dense(units=128, activation='relu')(flatten_tensor)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        return output_layer

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=[block1(input_layer), block2(block1(input_layer))])

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])