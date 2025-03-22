import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch for local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)
    
    # Second branch for downsampling and upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    transposed_conv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv2)
    
    # Concatenate the outputs of the branches
    concat = Concatenate()([avg_pool1, transposed_conv2])
    
    # Refine with a 1x1 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(conv3)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model