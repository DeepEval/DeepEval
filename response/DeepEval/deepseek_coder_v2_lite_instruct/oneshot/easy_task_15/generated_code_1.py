import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # 3x3 convolutional layer
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Two 1x1 convolutional layers
        conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv1x1_1)
        # Average pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1x1_2)
        # Dropout layer
        dropout = Dropout(0.5)(avg_pool)
        return dropout

    # Apply the block twice
    block1_output = block(input_layer)
    block2_output = block(block1_output)

    # Global average pooling layer
    global_avg_pool = AveragePooling2D(pool_size=(7, 7))(block2_output)
    flatten_layer = Flatten()(global_avg_pool)

    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)

    return model