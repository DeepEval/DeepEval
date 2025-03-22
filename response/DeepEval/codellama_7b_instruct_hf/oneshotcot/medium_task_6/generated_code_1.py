from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the initial convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the first parallel block
    conv2 = Conv2D(64, (1, 1), activation='relu')(conv1)
    batch_norm1 = BatchNormalization()(conv2)
    relu1 = ReLU()(batch_norm1)

    # Define the second parallel block
    conv3 = Conv2D(64, (3, 3), activation='relu')(conv1)
    batch_norm2 = BatchNormalization()(conv3)
    relu2 = ReLU()(batch_norm2)

    # Define the third parallel block
    conv4 = Conv2D(64, (5, 5), activation='relu')(conv1)
    batch_norm3 = BatchNormalization()(conv4)
    relu3 = ReLU()(batch_norm3)

    # Define the final convolutional layer
    conv5 = Conv2D(64, (1, 1), activation='relu')(conv1)
    batch_norm4 = BatchNormalization()(conv5)
    relu4 = ReLU()(batch_norm4)

    # Add the outputs of the parallel blocks
    output = Concatenate()([relu1, relu2, relu3, relu4])

    # Flatten the output
    flattened = Flatten()(output)

    # Define the fully connected layers
    dense1 = Dense(512, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model