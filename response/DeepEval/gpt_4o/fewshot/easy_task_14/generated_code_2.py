import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Global Average Pooling to capture global information
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers to generate weights matching the number of channels
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='relu')(dense1)

    # Reshape the weights to align with the input shape
    reshaped_weights = Reshape((1, 1, 3))(dense2)

    # Element-wise multiplication with the input feature map
    scaled_features = Multiply()([input_layer, reshaped_weights])

    # Flatten the result and pass through another fully connected layer
    flatten = Flatten()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data (for demonstration)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model (this is optional and for demonstration purposes)
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))