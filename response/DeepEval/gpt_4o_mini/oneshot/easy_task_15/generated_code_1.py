import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def feature_block(input_tensor):
        # 3x3 convolution
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Two 1x1 convolutions
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_2)
        # Dropout layer
        drop = Dropout(0.25)(avg_pool)

        return drop

    # First feature block
    block_output_1 = feature_block(input_layer)
    # Second feature block
    block_output_2 = feature_block(block_output_1)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(block_output_2)
    # Flattening layer
    flatten_layer = Flatten()(global_avg_pool)
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model