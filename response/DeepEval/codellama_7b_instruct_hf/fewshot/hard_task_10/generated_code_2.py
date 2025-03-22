import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

 和 return model
def dl_model():
    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Define the first path with a 1x1 convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)

    # Define the second path with a sequence of convolutions
    conv2 = Conv2D(filters=16, kernel_size=(1, 7), activation='relu')(input_layer)
    conv3 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu')(input_layer)

    # Merge the outputs of the two paths through addition
    merged = Add()([conv1, conv2, conv3])

    # Apply a 1x1 convolution to align the output dimensions with the input image's channel
    aligned = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(merged)

    # Define the branch that connects directly to the input
    branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Merge the outputs of the main path and the branch through addition
    merged_output = Add()([aligned, branch])

    # Flatten the output and pass it through two fully connected layers for classification
    flattened = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model