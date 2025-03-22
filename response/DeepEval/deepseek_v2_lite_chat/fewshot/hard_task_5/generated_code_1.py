import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split into three groups and reduce channels
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split[2])
        concat = Concatenate(axis=2)([conv1_1, conv1_2, conv1_3])
        return concat

    # Block 2: Channel shuffling and depthwise separable convolution
    def block2(input_tensor):
        shape = keras.backend.shape(input_tensor)
        new_shape = (shape[0], shape[1] // 4, 4, shape[3] * 3)
        input_tensor = keras.backend.reshape(input_tensor, new_shape)
        input_tensor = keras.backend.permute_dimensions(input_tensor, (0, 3, 1, 2))
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv2

    # Block 3: Additional feature extraction
    def block3(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        return conv3

    # Block 4: Final processing and classification
    def block4(input_tensor):
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        add_layer = Add()([input_tensor, conv4])
        flatten = Flatten()(add_layer)
        dense = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(dense)
        return output

    # Main path
    main_path = block1(input_layer)
    main_path = block2(main_path)
    main_path = block3(main_path)
    output_main = block4(main_path)

    # Branch
    branch_input = input_layer
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch_input)
    branch_output = block4(branch_path)

    # Combine main path and branch
    combined = Add()([output_main, branch_output])

    # Fully connected layer and output
    output = Dense(units=10, activation='softmax')(combined)
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)

    return model

# Create the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())