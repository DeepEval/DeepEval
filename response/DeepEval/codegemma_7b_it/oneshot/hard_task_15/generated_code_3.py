import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1)
    avg_pool = GlobalAveragePooling2D()(conv_2)
    dense_1 = Dense(units=128, activation='relu')(avg_pool)
    dense_2 = Dense(units=10, activation='softmax')(dense_1)

    # Branch path
    branch_conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv_1)

    # Combine main and branch paths
    concat = Add()([dense_2, branch_conv_2])

    # Final fully connected layers
    dense_3 = Dense(units=128, activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(dense_3)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model