import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main)
    global_avg_pool_main = GlobalAveragePooling2D()(conv_main)
    dense_main1 = Dense(units=64, activation='relu')(global_avg_pool_main)
    dense_main2 = Dense(units=32, activation='relu')(dense_main1)
    reshape_main = Reshape((1, 1, 32))(dense_main2)  # Reshape to match input shape
    multiply_main = Multiply()([input_layer, reshape_main])  # Element-wise multiplication

    # Branch Path
    branch = Input(shape=(32, 32, 3))  # Input to branch path

    # Combined Path
    combined = Add()([multiply_main, branch])

    # Final Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=[input_layer, branch], outputs=output_layer)

    return model