import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: three parallel paths
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(32, (2, 2), activation='relu')(input_layer)
    path3 = Conv2D(32, (4, 4), activation='relu')(input_layer)

    # Max pooling layers
    maxpool1 = MaxPooling2D((2, 2))(path1)
    maxpool2 = MaxPooling2D((2, 2))(path2)
    maxpool3 = MaxPooling2D((2, 2))(path3)

    # Flatten and concatenate
    flatten1 = Flatten()(maxpool1)
    flatten2 = Flatten()(maxpool2)
    flatten3 = Flatten()(maxpool3)
    output1 = Concatenate()([flatten1, flatten2, flatten3])

    # Block 2: four parallel paths
    path1 = Conv2D(64, (1, 1), activation='relu')(output1)
    path2 = Conv2D(64, (1, 7), activation='relu')(output1)
    path3 = Conv2D(64, (7, 1), activation='relu')(output1)
    path4 = Conv2D(64, (1, 1), activation='relu')(output1)

    # Max pooling layers
    maxpool1 = MaxPooling2D((2, 2))(path1)
    maxpool2 = MaxPooling2D((2, 2))(path2)
    maxpool3 = MaxPooling2D((2, 2))(path3)
    maxpool4 = MaxPooling2D((2, 2))(path4)

    # Flatten and concatenate
    flatten1 = Flatten()(maxpool1)
    flatten2 = Flatten()(maxpool2)
    flatten3 = Flatten()(maxpool3)
    flatten4 = Flatten()(maxpool4)
    output2 = Concatenate()([flatten1, flatten2, flatten3, flatten4])

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(output2)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Return the constructed model
    return model