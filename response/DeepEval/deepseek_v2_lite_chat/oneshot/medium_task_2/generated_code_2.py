import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize inputs to [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Input layers
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def conv_block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return pool

    # Branch path
    def branch_conv_block(input_tensor, filters):
        branch_conv = Conv2D(filters=filters, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_conv

    # Combine paths
    pool_main = conv_block(input_layer, 64)
    branch_conv = branch_conv_block(input_layer, 64)
    conv_output = Concatenate()([pool_main, branch_conv])

    # Batch normalization and flatten
    bn = BatchNormalization()(conv_output)
    flat = Flatten()(bn)

    # Dense layers
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Instantiate and train the model
model = dl_model()
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))