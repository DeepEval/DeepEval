import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Input data
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv)  # Integration path
    conv = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    drop = Lambda(lambda x: keras.backend.dropout(x, rate=0.5))(conv)  # Dropout layer

    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_conv = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)

    # Concatenate outputs
    merged = Concatenate()([conv, branch_conv])

    # Process merged output
    norm = BatchNormalization()(merged)
    flat = Flatten()(norm)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model