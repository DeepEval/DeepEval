import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Main path: two convolution and dropout blocks
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(0.2)(conv_block1)
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.2)(conv_block2)

    # Main path: restore the number of channels
    conv_restore = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop2)

    # Branch path: connect directly to the input
    branch_path = input_layer

    # Combine the outputs from both paths using addition
    adding_layer = Add()([conv_restore, branch_path])

    # Flattening layer and fully connected layer for classification
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model