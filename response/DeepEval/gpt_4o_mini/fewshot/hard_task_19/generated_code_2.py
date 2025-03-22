import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(main_conv2)
    main_pool = MaxPooling2D(pool_size=(2, 2))(main_conv3)

    # Branch path
    branch_avg_pool = GlobalAveragePooling2D()(input_layer)
    branch_dense1 = Dense(units=64, activation='relu')(branch_avg_pool)
    branch_dense2 = Dense(units=128, activation='sigmoid')(branch_dense1)  # Channel weights
    channel_weights = Reshape((1, 1, 128))(branch_dense2)  # Reshape for multiplication

    # Multiply the original input with channel weights
    multiplied = Multiply()([input_layer, channel_weights])

    # Combine the outputs from both paths
    added_output = Add()([main_pool, multiplied])

    # Final classification layers
    flatten = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model