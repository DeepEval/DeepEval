import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = BatchNormalization()(conv2)
    concat = Add()([pool2, input_layer])

    # Branch path
    branch = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Combine paths
    combined = Add()([concat, branch])
    combined = Flatten()(combined)
    dense1 = Dense(128, activation='relu')(combined)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])