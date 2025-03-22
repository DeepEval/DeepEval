import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.layers import Input, Concatenate

def dl_model():
    # Input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32
    
    # First block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(Input(shape=input_shape))
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Second block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Third block
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Parallel branch
    parallel_x = Conv2D(64, (3, 3), activation='relu')(Input(shape=input_shape))
    
    # Concatenate all outputs
    concat = Concatenate()([x, parallel_x])
    
    # Flatten and pass through fully connected layers
    x = Flatten()(concat)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10
    
    # Model
    model = Model(inputs=[Input(shape=input_shape), parallel_x], outputs=x)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])