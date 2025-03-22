import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)

    # Define the second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)

    # Define the third block
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3)
    block3 = BatchNormalization()(block3)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3)

    # Flatten the output of the third block
    flattened = Flatten()(block3)

    # Add two fully connected layers to produce the final classification probabilities
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model