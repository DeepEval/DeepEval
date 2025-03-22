import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    # Main Path
    block1 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block1 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)

    block2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1)
    block2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block2)

    # Summing outputs from main and branch paths
    output_tensor = keras.layers.Add()([block2, branch_path])

    # Flatten and Dense Layer
    batch_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model