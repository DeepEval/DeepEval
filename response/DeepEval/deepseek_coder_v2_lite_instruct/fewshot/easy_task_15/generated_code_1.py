import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        # 3x3 convolutional layer
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Two 1x1 convolutional layers
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(conv1x1_1)
        # Average pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1x1_2)
        # Dropout layer to prevent overfitting
        dropout = Dropout(0.25)(avg_pool)
        return dropout

    # Applying the specialized block twice
    block1_output = specialized_block(input_layer)
    block2_output = specialized_block(block1_output)

    # Global average pooling layer
    global_avg_pool = AveragePooling2D(pool_size=(7, 7))(block2_output)
    # Flattening layer
    flatten = Flatten()(global_avg_pool)
    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model