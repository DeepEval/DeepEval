from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.layers import LayerNormalization, Dropout

def dl_model():
    # Parameters
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels
    num_classes = 10  # There are 10 classes in MNIST

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1: Three parallel paths
    def block1():
        # Path 1: Average pooling of 1x1, 2x2, and 4x4 scales
        pool1 = MaxPooling2D((1, 1))(inputs)
        pool2 = MaxPooling2D((2, 2))(inputs)
        pool3 = MaxPooling2D((4, 4))(inputs)
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        drop1 = Dropout(0.5)(flat1)
        drop2 = Dropout(0.5)(flat2)
        drop3 = Dropout(0.5)(flat3)

        # Concatenate and fuse paths
        concat = Concatenate()([drop1, drop2, drop3])
        # ... Add a fully connected layer here

        # Path 2 and 3 follow similar structure
        # ...

        return Model(inputs=inputs, outputs=concat)

    block1_model = block1()

    # Block 2: Multiple branch connections
    def block2():
        # Branch 1: 1x1 convolution
        conv1_1 = Conv2D(16, (1, 1), activation='relu')(inputs)
        # Branch 2: 1x1 -> 3x3 convolution
        conv1_2 = Conv2D(16, (1, 1), activation='relu')(inputs)
        conv2_1 = Conv2D(16, (3, 3), activation='relu')(inputs)
        conv2_2 = Conv2D(16, (3, 3), activation='relu')(inputs)
        conv3_1 = Conv2D(16, (3, 3), activation='relu')(inputs)
        conv3_2 = Conv2D(16, (3, 3), activation='relu')(inputs)
        # Average pooling
        pool = MaxPooling2D((8, 8))(inputs)
        # Path 4: Average pooling -> 3x3 convolution -> 3x3 convolution
        pool_conv1 = Conv2D(32, (1, 1), activation='relu')(pool)
        conv4_1 = Conv2D(32, (3, 3), activation='relu')(pool_conv1)
        conv4_2 = Conv2D(32, (3, 3), activation='relu')(pool_conv1)

        # Concatenate and fuse branches
        concat1 = Concatenate()([conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2])
        concat2 = Concatenate()([pool_conv1, conv4_1, conv4_2])
        concat = Concatenate()([concat1, concat2])

        # Add fully connected layers
        dense1 = Dense(256, activation='relu')(concat)
        dense2 = Dense(128, activation='relu')(dense1)
        output = Dense(num_classes, activation='softmax')(dense2)

        return Model(inputs=inputs, outputs=output)

    block2_model = block2()

    # Final model
    model = Model(inputs=inputs, outputs=output)

    return model

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])