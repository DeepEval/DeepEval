import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Second branch
    conv2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)

    # Third branch
    conv3 = Conv2D(128, (1, 1), activation='relu')(input_layer)
    conv3 = Conv2D(128, (5, 5), activation='relu')(conv3)

    # Fourth branch
    pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_layer)
    conv4 = Conv2D(256, (1, 1), activation='relu')(pool)

    # Concatenate outputs
    merged = Add()([conv1, conv2, conv3, conv4])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(merged)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model


model.fit(cifar_train, epochs=10)


test_loss, test_acc = model.evaluate(cifar_test)
print('Test accuracy:', test_acc)