import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # First average pooling layer
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(batch_norm1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    batch_norm2 = BatchNormalization()(conv2)
    
    # Second average pooling layer
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    batch_norm3 = BatchNormalization()(conv3)
    
    # Third average pooling layer
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(batch_norm3)
    
    # Flatten the outputs of the average pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Flatten the concatenated output
    flatten_concat = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_concat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model