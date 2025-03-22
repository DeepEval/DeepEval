import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    avg_pool_main = GlobalAveragePooling2D()(conv_main)
    dense_main_1 = Dense(units=64, activation='relu')(avg_pool_main)
    dense_main_2 = Dense(units=32, activation='relu')(dense_main_1)
    reshape_main = Reshape(target_shape=(1, 1, 32))(dense_main_2)
    multiply_main = Multiply()([conv_main, reshape_main])

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)

    # Addition of main and branch paths
    added = Add()([multiply_main, conv_branch])

    # Final processing
    global_avg_pool = GlobalAveragePooling2D()(added)
    dense_final_1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense_final_2 = Dense(units=64, activation='relu')(dense_final_1)
    output_layer = Dense(units=10, activation='softmax')(dense_final_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model