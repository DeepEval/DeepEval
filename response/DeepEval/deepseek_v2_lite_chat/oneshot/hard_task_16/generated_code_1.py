import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for three parallel paths
    x = Lambda(lambda tensors: keras.backend.split(tensors, 3, axis=-1))(input_layer)
    
    # Three parallel paths for 1x1, 3x3, and 1x1 convolutions
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[0])
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x[1])
    path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[2])
    
    # Concatenate outputs from the three paths
    concatenated = Concatenate()(path1 + path2 + path3)
    
    # Transition layer to adjust channels
    transition = Conv2D(filters=x[0].shape[-1], kernel_size=(1, 1), padding='same')(x[0])
    
    # Block 1: Three convolutions and global max pooling
    block1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(transition)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    
    # Block 2: Two fully connected layers for channel weights, reshape, and multiply
    block2 = Flatten()(block1)
    block2 = Dense(units=512, activation='relu')(block2)
    weights = Dense(units=block1.shape[-1])(block2)  # Match weights size to input channels
    weights = keras.layers.Reshape((1, 1, -1))(weights)  # Reshape to match output shape
    output = keras.layers.multiply([weights, block1])  # Multiply with the adjusted output
    
    # Branch directly connected to input
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Combine outputs from main path and branch
    combined = Concatenate()([output, branch])
    
    # Final classification layer
    output = Dense(units=10, activation='softmax')(combined)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Create and display the model
model = dl_model()
model.summary()