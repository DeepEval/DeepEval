import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: three convolutional layers, max pooling
    def block1(input_tensor):
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        return conv3, pool3

    # Block 2: four convolutional layers, max pooling
    def block2(input_tensor):
        conv4 = Conv2D(256, (1, 1), activation='relu', padding='same')(input_tensor)
        conv4 = BatchNormalization()(conv4)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(input_tensor)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(256, (5, 5), activation='relu', padding='same')(input_tensor)
        conv6 = BatchNormalization()(conv6)
        conv7 = Conv2D(256, (1, 1), activation='relu', padding='same')(input_tensor)
        conv7 = BatchNormalization()(conv7)
        pool4 = MaxPooling2D(pool_size=(2, 2), padding='valid')(input_tensor)
        return Concatenate()([conv4, conv5, conv6, conv7, pool4])

    # Extract features from the second block
    conv3, pool3 = block1(input_layer)
    conv5, pool4 = block2(conv3)

    # Flatten and feed into fully connected layers
    flatten = Flatten()(pool4)
    dense1 = Dense(512, activation='relu')(flatten)
    dense2 = Dense(256, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()