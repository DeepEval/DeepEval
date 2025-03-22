import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups for the main path
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)

    # Main path
    def extract_features(input_tensor):
        # 1x1 depthwise separable conv
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor[0])
        # 3x3 depthwise separable conv
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor[1])
        # 5x5 depthwise separable conv
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor[2])

        # Concatenate the outputs from the three groups
        concatenated = Add()([conv1, conv2, conv3])
        return concatenated

    # Branch path
    def branch_features(input_tensor):
        # 1x1 convolutional layer
        conv = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    branch_output = branch_features(input_tensor=x[0])
    main_output = extract_features(input_tensor=x[1])
    branch_output = branch_features(input_tensor=x[2])

    # Add the outputs from the main and branch paths
    merged_output = Add()([main_output, branch_output])

    # Flatten and fully connected layers for classification
    flattened = Flatten()(merged_output)
    dense1 = Dense(units=512, activation='relu')(flattened)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])