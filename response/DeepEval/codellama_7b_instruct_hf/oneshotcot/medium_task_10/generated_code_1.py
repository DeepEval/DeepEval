import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first level of the residual connection structure
    basic_block = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(basic_block)
    output_block = Concatenate()([basic_block, batch_norm])

    # Define the second level of the residual connection structure
    residual_block1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_block)
    residual_block1 = BatchNormalization()(residual_block1)
    residual_block2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_block)
    residual_block2 = BatchNormalization()(residual_block2)
    output_block = Concatenate()([residual_block1, residual_block2])

    # Define the third level of the residual connection structure
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_block)
    output_block = Concatenate()([output_block, global_branch])

    # Define the output layer
    flatten = Flatten()(output_block)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model 