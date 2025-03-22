import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Block 2: Concatenate with the output of Block 1
    concat1 = Concatenate()([input_layer, relu1])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    # Block 3: Concatenate with the output of Block 2
    concat2 = Concatenate()([concat1, relu2])
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat2)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(relu3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))