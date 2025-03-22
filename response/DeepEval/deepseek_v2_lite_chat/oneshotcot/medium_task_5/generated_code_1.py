import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Add, BatchNormalization, Flatten, Dense, concatenate

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return maxpool

    main_output = block(input_layer)

    # Branch path
    def branch_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
        return maxpool

    branch_output = branch_block(input_tensor=input_layer)

    # Combine outputs from both paths
    combined_output = Add()([main_output, branch_output])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(batch_norm)

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()