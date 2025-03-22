import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        # 3x3 convolutional layer
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Two 1x1 convolutional layers
        conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
        # Average pooling layer
        avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_1x1_2)
        # Dropout layer to mitigate overfitting
        dropout_layer = Dropout(rate=0.25)(avg_pooling)

        return dropout_layer

    # Apply two consecutive specialized blocks
    block1_output = specialized_block(input_layer)
    block2_output = specialized_block(block1_output)

    # Global average pooling layer
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)

    # Flattening layer
    flatten_layer = Flatten()(global_avg_pooling)

    # Fully connected layer to produce final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model