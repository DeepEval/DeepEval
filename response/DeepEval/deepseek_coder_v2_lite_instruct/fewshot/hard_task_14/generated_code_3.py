import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    avg_pool = GlobalAveragePooling2D()(conv_main)
    dense_main = Dense(units=64, activation='relu')(avg_pool)
    weights_main = Dense(units=32, activation='sigmoid')(dense_main)
    reshaped_weights_main = Reshape((1, 1, 32))(weights_main)
    multiplied_main = Multiply()([conv_main, reshaped_weights_main])

    # Branch Path
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)

    # Merge Paths
    added = Add()([multiplied_main, conv_branch])

    # Output Path
    flatten_output = Flatten()(added)
    dense_output_1 = Dense(units=128, activation='relu')(flatten_output)
    dense_output_2 = Dense(units=64, activation='relu')(dense_output_1)
    output_layer = Dense(units=10, activation='softmax')(dense_output_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model