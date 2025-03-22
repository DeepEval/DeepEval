from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense
from keras.models import Model


def dl_model():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Input shapes
    input_shape = (28, 28, 1)

    # Main path
    input_main = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_main)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    dropout = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)

    # Branch path
    input_branch = Input(shape=input_shape)
    branch_x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_branch)
    branch_x = MaxPooling2D(pool_size=(2, 2))(branch_x)
    branch_x = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_x)
    branch_x = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_x)

    # Combine paths
    combined_x = Add()([dropout, branch_x])
    x = Flatten()(combined_x)
    output = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Return the model
    return model

    # Example usage:
    model = dl_model()
    model.fit([x_train, x_train], y_train, epochs=10, batch_size=64, validation_data=([x_test, x_test], y_test))