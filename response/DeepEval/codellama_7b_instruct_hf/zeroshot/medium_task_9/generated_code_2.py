from keras.applications import VGG16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Main structure of the model
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, inputs])
    
    # Feature fusion
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, x])
    
    # Average pooling layer
    x = AveragePooling2D((2, 2))(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=x)
    
    return model