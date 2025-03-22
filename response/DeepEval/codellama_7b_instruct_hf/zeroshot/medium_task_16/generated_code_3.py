from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Flatten, Dense
from keras.applications import VGG16


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the number of output classes
    num_classes = 10

    # Define the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Define the custom model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

    # Add the custom layers
    x = model.output
    x = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x[0])
    x = Conv2D(64, (3, 3), activation='relu')(x[1])
    x = Conv2D(64, (3, 3), activation='relu')(x[2])
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    return model