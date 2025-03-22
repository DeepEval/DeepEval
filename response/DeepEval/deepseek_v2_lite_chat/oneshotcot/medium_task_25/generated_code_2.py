import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    path1 = BatchNormalization()(path1)
    path1 = keras.activations.relu(path1)
    
    # Path 2: Average pooling followed by 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = keras.activations.relu(path2)
    
    # Path 3: 1x1 convolution followed by two parallel convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path3 = BatchNormalization()(path3)
    path3 = keras.activations.relu(path3)
    
    path3a = Conv2D(filters=32, kernel_size=(1, 3), padding='same')(input_layer)
    path3b = Conv2D(filters=32, kernel_size=(3, 1), padding='same')(input_layer)
    path3 = Concatenate()([path3, path3a, path3b])
    path3 = BatchNormalization()(path3)
    path3 = keras.activations.relu(path3)
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution
    path4 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path4 = BatchNormalization()(path4)
    path4 = keras.activations.relu(path4)
    
    path4a = Conv2D(filters=64, kernel_size=(1, 3), padding='same')(input_layer)
    path4b = Conv2D(filters=64, kernel_size=(3, 1), padding='same')(input_layer)
    path4 = Concatenate()([path4, path4a, path4b])
    path4 = BatchNormalization()(path4)
    path4 = keras.activations.relu(path4)
    
    # Concatenate all paths
    concat = Concatenate()([path1, path2, path3, path4])
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concat)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()