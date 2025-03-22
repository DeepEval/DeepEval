import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate branches
    merged_features = Concatenate()([branch1, branch2, branch3])

    # Adjust dimensions with 1x1 convolution
    merged_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged_features)

    # Add main path and branch
    main_path = Add()([input_layer, merged_features])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model