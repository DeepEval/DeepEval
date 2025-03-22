import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Conv2DTranspose, concatenate
from keras.layers import BatchNormalization, Activation

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = concatenate([x, x])
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = concatenate([x, x])
    
    x = Conv2D(3, (3, 3), activation='sigmoid')(x)
    
    # Model
    model = Model(inputs=input_layer, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Summary of the model
model.summary()