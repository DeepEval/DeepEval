import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        dropout = keras.layers.Dropout(rate=0.5)(max_pool)

        return dropout

    main_dropout = main_path(input_tensor=input_layer)
    flatten = Flatten()(main_dropout)

    # Branch pathway
    def branch_path(input_tensor):
        branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        return branch_conv

    branch_conv = branch_path(input_tensor=input_layer)

    # Concatenate outputs
    concat = Concatenate(axis=-1)([flatten, branch_conv])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', accuracy)