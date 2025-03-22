import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: Second set of Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Stage 3: Four parallel paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(maxpool2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool2)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(maxpool2)
    path4 = MaxPooling2D(pool_size=(1, 1), padding='same')(maxpool2)

    # Concatenate the outputs of the parallel paths
    concat = Concatenate()([path1, path2, path3, path4])

    # Stage 4: Batch Normalization and Flatten
    batchnorm = BatchNormalization()(concat)
    flatten = Flatten()(batchnorm)

    # Stage 5: Two sets of Convolution and UpSampling
    upconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(flatten)
    upconv2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(upconv1)

    # Skip connections to the corresponding layers
    skip_conn1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    skip_conn2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)

    upconv1 = keras.layers.add([upconv1, skip_conn1])
    upconv2 = keras.layers.add([upconv2, skip_conn2])

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(upconv2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)