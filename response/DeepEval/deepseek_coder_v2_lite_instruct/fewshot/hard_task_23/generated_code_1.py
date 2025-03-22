import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: Local feature extraction
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    
    # Second branch: Downsampling followed by 3x3 convolutional layer
    avg_pool2d_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2d_1)
    conv2_transpose = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv2_1)
    
    # Third branch: Downsampling followed by 3x3 convolutional layer
    avg_pool2d_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2d_2)
    conv3_transpose = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv3_1)
    
    # Concatenate the outputs of the branches
    concatenated = Concatenate()([conv1_2, conv2_transpose, conv3_transpose])
    
    # Refine the concatenated output with a 1x1 convolutional layer
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Flatten the refined output
    flattened = Flatten()(conv1x1)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model