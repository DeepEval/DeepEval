import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    separable_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 convolutional layer for feature extraction
    feat_extract = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(separable_conv)
    
    # Add dropout layers after each convolutional layer to prevent overfitting
    feat_extract = Dropout(rate=0.5)(feat_extract)
    
    # MaxPooling2D layer
    maxpooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(feat_extract)
    
    # Block with four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path3)
        return Concatenate()([path1, path2, path3, path4])
    
    block_output = block(maxpooling)
    block_output = BatchNormalization()(block_output)
    block_output = Flatten()(block_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(block_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model