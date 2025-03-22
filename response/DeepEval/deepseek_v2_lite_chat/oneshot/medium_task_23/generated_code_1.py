import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)
    
    # Path 2: 1x1 convolution followed by multiple convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1))(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 7))(path2, padding='same')
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    
    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), padding='valid')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    
    # Path 4: Average pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(1, 1))(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1))(path4)
    path4 = BatchNormalization()(path4)
    path4 = Activation('relu')(path4)
    
    # Concatenate the outputs of the four paths
    concat_layer = Concatenate()(list(range(4)) + [path4])
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()