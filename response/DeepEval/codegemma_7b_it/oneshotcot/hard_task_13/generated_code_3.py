import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Block 2: Global average pooling and fully connected layers
    gap = GlobalAveragePooling2D()(block1_output)
    flatten_layer = Flatten()(gap)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Load and preprocess CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

# # Train the model
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))