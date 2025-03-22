from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 Convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1: Local feature extraction with two sequential 3x3 convolutions
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Branch 2: Average pooling, 3x3 conv, transposed conv
    branch2 = AveragePooling2D((2, 2))(x)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)

    # Branch 3: Same as branch 2
    branch3 = AveragePooling2D((2, 2))(x)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Refinement 1x1 Convolutional layer
    refined = Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Flatten and Fully connected layer to produce classification result
    flatten = Flatten()(refined)
    output_layer = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model