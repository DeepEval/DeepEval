import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():     
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional layer 1x1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Convolutional layer 3x1
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(conv1)
    
    # Convolutional layer 1x3
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), padding='valid', activation='relu')(conv1)
    
    # Restore original channel count
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(conv3)
    
    # Add dropout to mitigate overfitting
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), padding='valid', activation='relu')(conv3)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    
    # Concatenate the outputs of the pooling layers
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concat)
    
    # Flatten layer
    flatten = Flatten()(batch_norm)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()