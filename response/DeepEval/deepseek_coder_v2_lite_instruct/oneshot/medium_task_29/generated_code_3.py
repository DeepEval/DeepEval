import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # First max pooling layer with 1x1 window
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    # Second max pooling layer with 2x2 window
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    # Third max pooling layer with 4x4 window
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(conv3)
    
    # Flatten the output from each pooling layer
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(batch_norm)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model