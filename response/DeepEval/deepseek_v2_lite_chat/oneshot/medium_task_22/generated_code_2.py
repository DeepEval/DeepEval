import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    
    # Branch 2: 1x1 Conv -> 3x3 Conv
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(inputs)
    
    # Concatenate branches
    concat = Concatenate()( [branch1, branch2, branch3] )
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    fc1 = Dense(units=128, activation='relu')(flatten)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output = Dense(units=10, activation='softmax')(fc2)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()