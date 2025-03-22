import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the input data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='sigmoid')(conv1)

    # Branch path
    branch_input = input_layer
    branch_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_input)
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='sigmoid')(branch_conv1)

    # Add the outputs of the main and branch paths
    main_output = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
    branch_output = Add()([conv1, branch_conv1])

    # Combine the outputs of both paths
    combined_output = Concatenate()([main_output, branch_output])

    # Pooling path
    pool1 = MaxPooling2D(pool_size=(1, 1), padding='valid')(combined_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(combined_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), padding='valid')(combined_output)

    # Flatten and connect to fully connected layers
    flat = Flatten()(pool3)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model