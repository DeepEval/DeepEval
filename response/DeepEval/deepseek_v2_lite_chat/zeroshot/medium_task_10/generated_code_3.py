from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D
from keras.optimizers import Adam

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Input shape
    input_shape = (32, 32, 3)

    # Input layer
    input_layer = Input(shape=input_shape)

    # First level residual block
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second level residual block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    branch_output = x

    # Third level residual block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    y = Conv2D(128, (3, 3), padding='same', name='branch_conv')(input_layer)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(128, (3, 3), padding='same', name='branch_conv_1')(input_layer)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    z = Add()([x, y])
    z = ReLU()(z)

    # Global branch
    x = Conv2D(128, (3, 3), padding='same', name='global_conv')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    global_output = x

    # Add global output to the second level output
    z = Add()([z, global_output])
    z = ReLU()(z)

    # Classification layer
    output = AveragePooling2D(pool_size=(4, 4))(z)
    output = GlobalAveragePooling2D()(output)
    y_pred = Dense(10, activation='softmax')(output)

    # Create the Keras model
    model = Model(inputs=input_layer, outputs=y_pred)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model