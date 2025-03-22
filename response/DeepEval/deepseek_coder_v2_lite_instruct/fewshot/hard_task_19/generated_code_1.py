import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    weights = Dense(128, activation='softmax')(branch_path)
    weights = Dense(128, activation='softmax')(weights)  # This is a bit redundant, but ensures the weights are normalized
    weights = keras.layers.Reshape((1, 1, 128))(weights)

    # Multiply weights with the input
    weighted_input = Multiply()([input_layer, weights])

    # Add outputs from both paths
    added_output = Add()([main_path, weighted_input])

    # Additional fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model