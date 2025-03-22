import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def dl_model():
    # Input layers
    input_img = Input(shape=(32, 32, 3,))
    input_aux = Input(shape=(32, 32, 3,))
    
    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x1 = MaxPooling2D((2, 2))(x)
    
    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x2 = MaxPooling2D((2, 2))(x)
    
    # Third convolutional layer
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x3 = MaxPooling2D((2, 2))(x)
    
    # Separate convolutional layer for input image
    x_in = Conv2D(64, (3, 3), activation='relu', padding='same')(input_aux)
    
    # Concatenate features from all convolutional layers
    concat = concatenate([x3, x, x_in])
    
    # Flatten and pass through two fully connected layers
    x = Flatten()(concat)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=[input_img, input_aux], outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model