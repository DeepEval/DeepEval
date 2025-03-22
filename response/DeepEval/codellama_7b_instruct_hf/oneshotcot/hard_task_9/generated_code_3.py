import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first feature extraction branch
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Define the second feature extraction branch
    branch2 = Conv2D(32, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Define the third feature extraction branch
    branch3 = Conv2D(32, (1, 1), activation='relu')(branch2)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Add a 1x1 convolution layer to adjust the output dimensions
    merged = Conv2D(32, (1, 1), activation='relu')(merged)

    # Add batch normalization and flatten layers
    merged = BatchNormalization()(merged)
    merged = Flatten()(merged)

    # Add three fully connected layers for classification
    merged = Dense(64, activation='relu')(merged)
    merged = Dense(64, activation='relu')(merged)
    output_layer = Dense(10, activation='softmax')(merged)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model