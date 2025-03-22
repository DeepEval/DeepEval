import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        return pool2

    main_output = main_path(input_tensor=input_layer)

    # Branch path
    def branch_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(bn)
        return pool

    branch_output = branch_path(input_tensor=input_layer)

    # Combine the outputs of both paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and project onto 10 classes
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)