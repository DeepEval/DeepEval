import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # First max pooling layer with 1x1 window
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(batch_norm1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    batch_norm2 = BatchNormalization()(conv2)
    
    # Second max pooling layer with 2x2 window
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(batch_norm2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    batch_norm3 = BatchNormalization()(conv3)
    
    # Third max pooling layer with 4x4 window
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(batch_norm3)
    
    # Flatten the output from the three pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concatenated)
    batch_norm_fc = BatchNormalization()(dense1)
    dense2 = Dense(units=128, activation='relu')(batch_norm_fc)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model