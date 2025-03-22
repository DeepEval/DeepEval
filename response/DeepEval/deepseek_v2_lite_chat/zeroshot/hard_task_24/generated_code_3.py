import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.applications import VGG16

# Load data and preprocess
from keras.datasets import cifar10

# Import necessary modules
from keras.utils import to_categorical
from keras.optimizers import Adam

def dl_model():
    # Load and preprocess CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Input layer
    input_img = Input(shape=(32, 32, 3,))
    
    # First branch: Local feature extraction with 3x3 conv + 1x1 conv
    x1 = Conv2D(64, (3, 3), activation='relu')(input_img)
    x1 = Conv2D(64, (1, 1), activation='relu')(x1)
    
    # Second branch: Max pooling + 3x3 conv + upsampling
    x2 = MaxPooling2D(pool_size=(3, 3))(input_img)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = UpSampling2D(size=(2, 2))(x2)
    
    # Third branch: Max pooling + 3x3 conv + upsampling
    x3 = MaxPooling2D(pool_size=(3, 3))(input_img)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = UpSampling2D(size=(2, 2))(x3)
    
    # Concatenate and pass through 1x1 conv for fusion
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Conv2D(64, (1, 1), activation='relu')(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Model
    model = Model(inputs=input_img, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and return the model
model = dl_model()