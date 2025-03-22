import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    
    # MaxPooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten and fully connected layers
    flatting = Flatten()(maxpool)
    dense1 = Dense(units=128, activation='relu')(flatting)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()