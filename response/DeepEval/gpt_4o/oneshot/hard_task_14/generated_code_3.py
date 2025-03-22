import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # CIFAR-10 input shape
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    input_layer = Input(shape=input_shape)

    # Main path
    gap = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=128, activation='relu')(gap)
    dense2_main = Dense(units=input_shape[-1], activation='sigmoid')(dense1_main)  # generate weights
    weights_reshaped = Reshape((1, 1, input_shape[-1]))(dense2_main)
    scaled_features = Multiply()([input_layer, weights_reshaped])

    # Branch path
    conv_branch = Conv2D(filters=input_shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined_output = Add()([scaled_features, conv_branch])

    # Fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=num_classes, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Optionally load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Optionally train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))