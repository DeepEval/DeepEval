import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, DepthwiseConv2D, Reshape

 和 return model
def dl_model():
    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the first block with three convolutional layers with varying kernel sizes
    conv1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv3 = Conv2D(128, kernel_size=(5, 5), activation='relu')(split_layer[2])

    # Concatenate the outputs from the three convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Apply dropout to reduce overfitting
    dropout = Dropout(0.2)(concatenated)

    # Define the second block with four branches
    branch1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(dropout)
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(dropout)
    branch3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(dropout)
    branch4 = DepthwiseConv2D(kernel_size=(7, 7), activation='relu')(dropout)

    # Concatenate the outputs from the four branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Define the fully connected layer with softmax activation
    output_layer = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model