import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = tf.split(input_layer, num_or_size_splits=3, axis=3)
    x1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0]))
    x2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1]))
    x3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2]))
    x = Concatenate()([x1, x2, x3])

    # Second block
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Lambda(lambda x: Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(x)
    x = Lambda(lambda x: Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(x)
    x = Lambda(lambda x: Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    # Final layers
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# # Preprocess data
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# # Train the model
# model.fit(x_train, y_train, epochs=10)

# # Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
# print('Loss:', loss)
# print('Accuracy:', accuracy)