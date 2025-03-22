import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Add convolutional paths
    merged = Add()([input_layer, pool])
    
    # Flattening
    flat = Flatten()(merged)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()