from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Add a 1x1 convolutional layer to increase dimensionality
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Add a 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseSeparableConv2D((3, 3), activation='relu')(x)

    # Add another 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(16, (1, 1), activation='relu')(x)

    # Add a max pooling layer to downsample the spatial dimensions
    x = MaxPooling2D((2, 2))(x)

    # Flatten the output
    x = Flatten()(x)

    # Add a fully connected layer for classification
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model