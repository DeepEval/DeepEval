import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first feature extraction path
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second feature extraction path
    conv1x7 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv7x1 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat = Concatenate()([conv1x7, conv7x1])

    # Define the main path
    conv1x1_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Define the branch
    branch = Add()([conv1x1_main, input_layer])

    # Define the final feature extraction path
    conv1x1_final = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch)

    # Define the flatten layer
    flatten = Flatten()(conv1x1_final)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model