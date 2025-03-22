from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Input shape
input_shape = (28, 28, 1)  # Adjust based on the actual input image shape

def dl_model():
    # Branch 1 and 2 with shared block
    inputs = Input(shape=input_shape)
    
    # Block (shared between branches)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Branch 1
    branch1_x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch1_x = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1_x)
    branch1_x = MaxPooling2D(pool_size=(2, 2))(branch1_x)
    
    # Branch 2
    branch2_x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch2_x = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2_x)
    branch2_x = MaxPooling2D(pool_size=(2, 2))(branch2_x)
    
    # Concatenate the outputs of the branches and the block
    concat_layer = Concatenate(axis=-1)([x, branch1_x, branch2_x])
    
    # Flatten and add fully connected layers
    x = Flatten()(concat_layer)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST
    
    # Model
    model = Model(inputs=inputs, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and return the model
model = dl_model()