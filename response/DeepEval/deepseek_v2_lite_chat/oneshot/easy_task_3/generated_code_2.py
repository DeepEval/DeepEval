import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Two convolution, max pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 2: Two convolution, convolution, max pooling layers
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Block 3: Two convolution, convolution, convolution, max pooling layers
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv5)
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv6)
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    # Concatenate features from all blocks
    concat = Concatenate()([max_pool1, max_pool2, max_pool3, max_pool4])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()