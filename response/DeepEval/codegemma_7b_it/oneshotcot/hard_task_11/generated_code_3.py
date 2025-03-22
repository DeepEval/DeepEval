import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Parallel branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate the branches
    concat_path = Concatenate()([main_path, branch1, branch2, branch3])

    # 1x1 convolution to reduce channels
    concat_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_path)

    # Direct connection from input
    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion operation
    output_layer = Add()([concat_path, shortcut])
    output_layer = BatchNormalization()(output_layer)

    # Fully connected layers
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=64, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model