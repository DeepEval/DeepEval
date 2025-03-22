import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Path 2: 1x1 convolution, 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path2)
    
    # Path 3: 1x1 convolution with multi-directional convolutions
    path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(path3)
    
    # Path 4: Average pooling, 1x1 convolution
    path4 = AveragePooling2D(pool_size=(1, 1))(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)
    
    # Concatenate the outputs of the paths
    fuse_layer = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Batch normalization, Flatten, and Dense layers
    bath_norm = BatchNormalization()(fuse_layer)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and print the model
model = dl_model()
model.summary()