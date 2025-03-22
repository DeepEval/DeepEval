import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv1 = LeakyReLU()(conv1)
    
    # 1x1 convolutional layer to match the channel count
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1)
    conv2 = LeakyReLU()(conv2)
    
    # 3x3 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
    conv3 = LeakyReLU()(conv3)
    
    # 1x3 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), padding='valid')(conv3)
    conv4 = LeakyReLU()(conv4)
    
    # Dropout layer to mitigate overfitting
    conv4 = Dropout(0.5)(conv4)
    
    # Concatenate the features from the different convolutional layers
    concat_layer = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat_layer)
    bn = LeakyReLU()(bn)
    flatten = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model