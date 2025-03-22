import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras.models import Model

def dl_model():     
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch_pool = GlobalAveragePooling2D()(branch_conv)
    branch_dense1 = Dense(units=64, activation='relu')(branch_pool)
    branch_dense2 = Dense(units=10)(branch_dense1)
    branch_output = Reshape((10, 1))(branch_dense2)

    # Multiply channel weights with the input
    channel_weights = Multiply()([branch_output, max_pooling])
    channel_weights = Reshape((32, 32, 64))(channel_weights)

    # Add main and branch paths
    added_output = Add()([conv3, channel_weights])

    # Classification path
    dense1 = Dense(units=128, activation='relu')(added_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model