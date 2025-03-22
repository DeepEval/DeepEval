import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Convolution2D
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: Additional Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Skip connection for spatial information restoration
    skip_conn1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(skip_conn1)

    # Skip connection for spatial information restoration
    skip_conn2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(skip_conn2)

    # Flatten and pass through two dense layers
    flat = Flatten()(pool2)
    dense1 = Dense(units=128, activation='relu')(flat)
    drop1 = Dense(units=64, activation='relu')(dense1)
    drop1 = keras.layers.Dropout(rate=0.5)(drop1)

    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(drop1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()