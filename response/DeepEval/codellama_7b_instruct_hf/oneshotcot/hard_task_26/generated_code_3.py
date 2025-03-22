import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv)

    # Branch 1
    branch1_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_pool)
    branch1_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1_conv)

    # Branch 2
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_pool)
    branch2_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)

    # Branch 3
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_pool)
    branch3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([branch1_conv, branch2_conv, branch3_conv])

    # Add a 1x1 convolutional layer to produce the final output
    final_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Flatten the output and add two fully connected layers for classification
    flattened = Flatten()(final_conv)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model