import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to the range of 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)

    # Direct branch for comparison
    direct_conv = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input_layer)

    # Concatenate all outputs
    concat_layer = Concatenate()([pool3, direct_conv])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)