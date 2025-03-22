import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Add, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutional layer followed by max pooling
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    branch1 = MaxPool2D(pool_size=(2, 2))(branch1)
    
    # Branch 2: 5x5 convolutional layer followed by max pooling
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    branch2 = MaxPool2D(pool_size=(2, 2))(branch2)
    
    # Add branches
    add_layer = Add()([branch1, branch2])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(add_layer)
    
    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(avg_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Softmax classifier
    outputs = Dense(units=10, activation='softmax')(fc2)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()