import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have size 32x32 and 3 color channels

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined = Concatenate()([main_path, branch_path])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model