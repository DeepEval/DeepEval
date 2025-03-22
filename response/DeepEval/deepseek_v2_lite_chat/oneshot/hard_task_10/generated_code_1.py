import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction path 1: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)
    
    # Feature extraction path 2
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(input_layer)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    
    # Concatenate paths
    concat_layer = Concatenate()([path1, path2])
    
    # 1x1 convolution to match output channel with input image
    conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(concat_layer)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(conv1x1)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model configuration
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()