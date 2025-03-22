import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, BatchNormalization, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch = input_layer
    
    # Add main and branch paths
    add = Add()([conv2, branch])
    
    # Second block
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(add)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(add)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(add)
    
    # Flatten outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the outputs
    concat = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model