import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU, SeparableConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = input_layer
    for i in range(2):
        main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(main_path)

    # Branch path
    branch_path = input_layer
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Merge main and branch paths
    merged_path = Add()([main_path, branch_path])

    # Flatten and classify
    flattened = Flatten()(merged_path)
    output = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model