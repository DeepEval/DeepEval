import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 64))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    main_path = Concatenate()([conv2_1, conv2_2])

    # Branch Path
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = conv3

    # Combine paths
    merged_path = Add()([main_path, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model