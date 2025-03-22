import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.applications import VGG16


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def dl_model():
    # Input layer
    input_main = Input(shape=(32, 32, 3))
    
    # Main path layers
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_main)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Branch path
    branch = GlobalAveragePooling2D()(pool)
    dense1 = Dense(512, activation='relu')(branch)
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Combine outputs of main and branch paths
    output = concatenate([conv3, dense2])
    
    # Final model
    model = Model(inputs=[input_main], outputs=[output])
    return model


model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)