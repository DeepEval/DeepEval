import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Reshape

def dl_model():
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Branch path
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Merge the main path and branch path
    merged = Add()([avg_pool, branch_path])

    # First block
    block1_output = merged

    # Second block
    global_avg_pool = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Reshape(target_shape=(64,))(dense2)

    # Multiply the weights with the input
    multiplied = Multiply()([block1_output, weights])

    # Flatten and classification
    flatten = Flatten()(multiplied)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model