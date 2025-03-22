import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool

    main_path = block(input_layer)

    # Branch path
    def branch_block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool

    branch_path = branch_block(input_tensor=input_layer)

    # Combine paths
    combined_output = Add()[(2, 2)]([main_path, branch_path])

    # Flatten and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model