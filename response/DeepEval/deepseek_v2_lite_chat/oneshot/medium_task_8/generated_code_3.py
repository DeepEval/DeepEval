import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Main path
    # First group remains unchanged
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(group1)
    bn1 = BatchNormalization()(conv1)
    relu1 = keras.layers.Activation('relu')(bn1)
    
    # Second group undergoes feature extraction
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(group2)
    bn2 = BatchNormalization()(conv2)
    relu2 = keras.layers.Activation('relu')(bn2)
    
    # Third group
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(group3)
    bn3 = BatchNormalization()(conv3)
    relu3 = keras.layers.Activation('relu')(bn3)
    
    # Combine outputs of all three groups
    concat = Concatenate(axis=-1)([relu1, relu2, relu3])
    
    # Additional 3x3 convolution
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat)
    bn4 = BatchNormalization()(conv4)
    relu4 = keras.layers.Activation('relu')(bn4)
    
    # Branch path
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)
    bn5 = BatchNormalization()(conv5)
    relu5 = keras.layers.Activation('relu')(bn5)
    
    # Combine main path and branch path outputs
    combined = Concatenate(axis=-1)([conv4, relu4, conv5, relu5])
    
    # Flatten and pass through fully connected layers
    flat = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model