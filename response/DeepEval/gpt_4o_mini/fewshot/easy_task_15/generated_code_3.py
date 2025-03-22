import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        pooled = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1_2)
        return pooled

    # First specialized block
    block1_output = specialized_block(input_layer)
    # Dropout to mitigate overfitting
    block1_output = Dropout(rate=0.25)(block1_output)

    # Second specialized block
    block2_output = specialized_block(block1_output)
    block2_output = Dropout(rate=0.25)(block2_output)

    # Global average pooling layer
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)

    # Flattening layer
    flatten_layer = Flatten()(global_avg_pooling)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model