import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    
    block2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)
    
    # Branch path
    branch = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch = MaxPooling2D(pool_size=(2, 2))(branch)
    
    # Combine paths
    combined = Add()([block2, branch])
    
    # Flatten and fully connected layers
    flat = Flatten()(combined)
    output = Dense(10, activation='softmax')(flat)
    
    # Model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()