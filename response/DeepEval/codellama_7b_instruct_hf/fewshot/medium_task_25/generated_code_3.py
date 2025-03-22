import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the four parallel branches
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = Conv2D(32, (1, 1), activation='relu')(path2)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = Conv2D(32, (1, 1), activation='relu')(path3)
    path4 = Conv2D(32, (1, 1), activation='relu')(path2)
    path4 = Conv2D(32, (3, 3), activation='relu')(path4)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of the parallel branches
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated outputs
    flattened = Flatten()(concatenated)

    # Add a fully connected layer for classification
    output = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    return model