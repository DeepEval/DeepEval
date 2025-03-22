import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Four branches
    def branch1(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    def branch2(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        return conv2
    
    def branch3(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv)
        return conv2
    
    def branch4(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(conv)
        return conv2

    # Concatenate the outputs of the branches
    output_tensor1 = branch1(input_layer)
    output_tensor2 = branch2(input_layer)
    output_tensor3 = branch3(input_layer)
    output_tensor4 = branch4(input_layer)
    output_tensor = Concatenate()([output_tensor1, output_tensor2, output_tensor3, output_tensor4])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(output_tensor)
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)