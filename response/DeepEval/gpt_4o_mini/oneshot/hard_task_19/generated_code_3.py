import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB)
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(main_path)

    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=32, activation='relu')(branch_path)
    channel_weights = Reshape((1, 1, 32))(branch_path)

    # Multiply the channel weights with the input
    weighted_input = Multiply()([input_layer, channel_weights])

    # Add the outputs from both paths
    added_output = Add()([main_path, weighted_input])

    # Fully connected layers for classification
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model