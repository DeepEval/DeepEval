import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1_2)
        return avg_pool

    # First specialized block
    block1_output = specialized_block(input_layer)
    dropout1 = Dropout(rate=0.3)(block1_output)

    # Second specialized block
    block2_output = specialized_block(dropout1)
    dropout2 = Dropout(rate=0.3)(block2_output)

    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(dropout2)

    # Fully connected layers
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model