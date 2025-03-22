import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    main_path = Add()([conv1, conv2, conv3])

    # Branch path
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Add()([conv4, conv3, conv2, conv1])

    # Combine main and branch paths
    merging_layer = Add()([main_path, branch_path])

    # Flatten and classify
    flatten_layer = Flatten()(merging_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model