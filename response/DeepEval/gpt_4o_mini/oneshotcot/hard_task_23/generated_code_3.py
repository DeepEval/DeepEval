import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Step 3: Define the first branch with two sequential 3x3 convolutional layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Step 4: Define the second branch
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch2)

    # Step 5: Define the third branch
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch3)

    # Step 6: Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Step 7: Add a 1x1 convolutional layer to refine the output
    refined_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(concatenated)

    # Step 8: Flatten the result
    flatten_layer = Flatten()(refined_output)

    # Step 9: Add the fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model