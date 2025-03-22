import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block1(input_tensor):
        # Path 1: 1x1 convolution
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        # Path 2: 3x3 depthwise separable convolution
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', use_depthwise=True)(input_tensor)
        # Path 3: 1x1 convolution
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3x3)
        # Branch Path 1: 3x3 depthwise separable convolution
        conv3x3_branch = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', use_depthwise=True)(input_tensor)
        # Branch Path 2: 1x1 convolution
        conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3x3_branch)
        # Concatenate the paths along the channel dimension
        concat_layer = Concatenate(axis=-1)([conv1x1, conv1x1_2, conv3x3_branch, conv1x1_branch, conv1x1_branch])
        return concat_layer

    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = keras.backend.int_shape(input_tensor)
        # Reshape into groups for channel shuffling
        input_tensor = keras.backend.reshape(input_tensor, shape=(shape[1] * shape[2], shape[3], shape[0], 4))
        input_tensor = keras.backend.permute_dimensions(input_tensor, pattern=(0, 2, 3, 1))
        input_tensor = keras.backend.reshape(input_tensor, shape=(shape[1], shape[2], shape[3]))
        return input_tensor

    input_layer = Input(shape=(28, 28, 1))
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])