import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1_2)
        dropout = Dropout(rate=0.25)(avg_pool)
        return dropout

    # Create two consecutive specialized blocks
    block1_output = specialized_block(input_layer)
    block2_output = specialized_block(block1_output)

    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)

    # Flatten and fully connected layer for output
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model