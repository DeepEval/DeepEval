import keras
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path2)
    
    # Path 3: 1x1 convolution followed by two parallel convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path3_conv1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path3)
    path3_conv2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path3)
    path3 = Concatenate(axis=-1)([path3_conv1, path3_conv2])
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path4)
    
    # Concatenate the outputs of the four paths
    concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concatenated)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()