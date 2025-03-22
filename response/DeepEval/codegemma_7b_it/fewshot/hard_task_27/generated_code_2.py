import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Spatial Feature Extraction
    conv_dw = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', use_bias=False)(input_layer)
    conv_dw = BatchNormalization()(conv_dw)

    # Fully Connected Channel Transformation
    flatten_a = Flatten()(conv_dw)
    dense_a = Dense(units=32, activation='relu')(flatten_a)
    dense_a = Dense(units=32, activation='relu')(dense_a)

    # Channel-wise Feature Transformation
    flatten_b = Flatten()(conv_dw)
    dense_b = Dense(units=32, activation='relu')(flatten_b)
    dense_b = Dense(units=32, activation='relu')(dense_b)

    # Combine Features
    concat = Add()([dense_a, dense_b])

    # Classification
    flatten_c = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten_c)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)