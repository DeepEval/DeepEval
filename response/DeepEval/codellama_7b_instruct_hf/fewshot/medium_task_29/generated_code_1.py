import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first max pooling layer with window size 1x1 and stride 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Define the second max pooling layer with window size 2x2 and stride 2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(pool1)

    # Define the third max pooling layer with window size 4x4 and stride 4
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(pool2)

    # Flatten the output of the third max pooling layer
    flatten = Flatten()(pool3)

    # Define the first fully connected layer with 64 units
    dense1 = Dense(units=64, activation='relu')(flatten)

    # Define the second fully connected layer with 32 units
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Define the output layer with 10 units and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model