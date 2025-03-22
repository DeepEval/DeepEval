import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model
    def create_branch(input_tensor, filters):
        conv1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        return Concatenate()([conv2, input_tensor])

    def create_model():
        input_layer = Input(shape=x_train.shape[1:])
        branch1 = create_branch(input_tensor=input_layer, filters=64)
        branch2 = create_branch(input_tensor=input_layer, filters=64)
        branch3 = create_branch(input_tensor=input_layer, filters=64)
        branch3 = Conv2D(x_train.shape[3], (1, 1), padding='same', activation='relu')(branch3)
        concat = Concatenate()(outputs=[branch1, branch2, branch3])
        conv = Conv2D(x_train.shape[3], (1, 1), padding='same', activation='relu')(concat)
        flatten = Flatten()(conv)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    return create_model()

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)