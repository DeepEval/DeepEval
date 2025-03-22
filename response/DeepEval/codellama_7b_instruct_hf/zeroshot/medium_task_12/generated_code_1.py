from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.applications.cifar10 import Cifar10Data



def dl_model():
    # Load the CIFAR-10 dataset
    cifar10 = Cifar10Data(input_shape=(32, 32, 3))

    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Define the first block
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Define the second block
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Define the third block
    x = Conv2D(2048, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(4096, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Define the output layer
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(cifar10.train.images, cifar10.train.labels, epochs=10, batch_size=32)

    return model