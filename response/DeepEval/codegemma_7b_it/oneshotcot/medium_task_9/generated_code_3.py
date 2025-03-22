import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Basic block 1
    def basic_block(input_tensor):
        # Main path
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm2 = BatchNormalization()(conv2)
        relu2 = ReLU()(batch_norm2)

        # Branch path
        conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm3 = BatchNormalization()(conv3)
        relu3 = ReLU()(batch_norm3)

        # Feature fusion
        add_output = Add()([relu2, relu3])
        relu4 = ReLU()(add_output)

        return relu4

    # Basic block 2
    basic_block_output = basic_block(input_tensor=relu1)

    # Average pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(basic_block_output)

    # Flatten
    flatten = Flatten()(avg_pool)

    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()