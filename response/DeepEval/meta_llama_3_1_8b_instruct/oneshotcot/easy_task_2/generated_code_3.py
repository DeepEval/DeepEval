import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(224, 224, 3))
    
    # Sequential feature extraction layers
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional convolutional layers and average pooling
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Define a block with three convolutional layers and average pooling
    def block(input_tensor):
        path1 = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    block_output = block(avg_pool3)
    avg_pool4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(block_output)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(avg_pool4)
    
    # Process through fully connected layers with dropout
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    # Output classification probabilities
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model