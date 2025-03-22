import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D
from keras.models import Model
from keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the input into three groups along the last dimension
    split_dim = 3
    x_train_split = Lambda(lambda x: keras.backend.split(x, split_dim, axis=-1))(x_train)
    x_test_split = Lambda(lambda x: keras.backend.split(x, split_dim, axis=-1))(x_test)

    # Define the first block
    def block1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate(axis=-1)([path1, path2, path3])
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Activation('relu')(output_tensor)
        output_tensor = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output_tensor)
        return output_tensor

    # Define the second block with multiple branches
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path7 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4, path5, path6, path7])
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Flatten()(output_tensor)
        output_tensor = Dense(units=128, activation='relu')(output_tensor)
        output_tensor = Dense(units=64, activation='relu')(output_tensor)
        output_tensor = Dense(units=10, activation='softmax')(output_tensor)
        return output_tensor

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    block1_output = block1(input_tensor=input_layer)
    block1_output = block2(input_tensor=block1_output)

    # Model compilation
    model = Model(inputs=input_layer, outputs=block1_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)