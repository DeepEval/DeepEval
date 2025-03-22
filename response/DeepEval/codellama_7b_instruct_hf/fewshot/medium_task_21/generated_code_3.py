import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def dl_model():
    # Define the input shape and input layer
    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape)

    # Define the four branches
    branch1 = Conv2D(32, kernel_size=1, activation='relu')(input_layer)
    branch1 = Conv2D(64, kernel_size=3, activation='relu')(branch1)

    branch2 = Conv2D(32, kernel_size=1, activation='relu')(input_layer)
    branch2 = Conv2D(64, kernel_size=3, activation='relu')(branch2)

    branch3 = Conv2D(32, kernel_size=1, activation='relu')(input_layer)
    branch3 = Conv2D(64, kernel_size=3, activation='relu')(branch3)

    branch4 = MaxPooling2D(pool_size=2, strides=2)(input_layer)
    branch4 = Conv2D(64, kernel_size=1, activation='relu')(branch4)

    # Define the concatenate layer
    concat_layer = keras.layers.concatenate([branch1, branch2, branch3, branch4])

    # Define the dropout layers
    dropout_layer1 = Dropout(0.2)(concat_layer)
    dropout_layer2 = Dropout(0.2)(dropout_layer1)

    # Define the flatten layer
    flatten_layer = Flatten()(dropout_layer2)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model