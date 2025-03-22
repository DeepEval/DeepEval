import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Pathway 1
    path1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    path1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1_conv1)
    path1_avg_pool = AveragePooling2D(pool_size=(2, 2))(path1_conv2)

    # Pathway 2
    path2_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine pathways using addition
    combined = Add()([path1_avg_pool, path2_conv])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model