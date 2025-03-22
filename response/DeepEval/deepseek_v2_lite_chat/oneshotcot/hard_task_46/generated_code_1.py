import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups for different kernel sizes
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # First block
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor[0])  # 1x1 kernel
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor[1])  # 3x3 kernel
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_tensor[2])  # 5x5 kernel
        concat = Concatenate()([conv1, conv2, conv3])
        return concat

    block1_output = first_block(split1)
    block1_bn = BatchNormalization()(block1_output)
    block1_flat = Flatten()(block1_bn)

    # Second block
    def second_block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)  # 3x3 kernel
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)  # 1x1 kernel
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)  # 3x3 kernel
        conv4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)  # MaxPooling
        concat = Concatenate()([conv2, conv3, conv4])
        return concat

    block2_output = second_block(block1_flat)
    block2_bn = BatchNormalization()(block2_output)
    block2_flat = Flatten()(block2_bn)
    global_pool = GlobalAveragePooling2D()(block2_flat)

    # Output layer
    dense1 = Dense(units=128, activation='relu')(global_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()
model.summary()