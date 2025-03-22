import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 dataset (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))

    # First block: feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Adding the input to the output of the first block
    residual_connection = Add()([input_layer, avg_pool])

    # Second block: global average pooling and fully connected layers
    global_avg_pool = GlobalAveragePooling2D()(residual_connection)
    channel_weights = Dense(units=32, activation='relu')(global_avg_pool)
    channel_weights = Dense(units=32, activation='sigmoid')(channel_weights)

    # Reshape to match the channels of the input for multiplication
    channel_weights_reshaped = Reshape((1, 1, 32))(channel_weights)

    # Multiply the input with the channel weights
    weighted_input = Multiply()([residual_connection, channel_weights_reshaped])

    # Flatten the weighted input for the final classification layer
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model