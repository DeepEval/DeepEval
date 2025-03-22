import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    concatenated_layer = Concatenate()([conv1, conv2, conv3])

    # Define the second block
    input_branch1 = Input(shape=input_shape)
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_branch1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
    concatenated_branch_layer = Concatenate()([branch1, branch2, branch3])

    # Define the final layer
    flattened_layer = Flatten()(concatenated_layer)
    output_layer = Dense(units=10, activation='softmax')(flattened_layer)

    # Create the model
    model = keras.Model(inputs=[input_layer, input_branch1], outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model