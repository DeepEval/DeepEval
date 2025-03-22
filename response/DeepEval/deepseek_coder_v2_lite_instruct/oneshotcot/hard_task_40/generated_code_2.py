import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer (optional, depending on how you want to start)
    # For this example, we'll start with average pooling layers
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Concatenate the outputs of the average pooling layers
    concat_pooling = Concatenate()([pool1, pool2, pool3])
    
    # Flatten the concatenated output
    flattened = Flatten()(concat_pooling)
    
    # Reshape the flattened output into a 4-dimensional tensor
    reshaped = Reshape((4,))(flattened)
    
    # Second block
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    
    # Path 2: 1x1 followed by two 3x3 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 followed by a single 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    
    # Path 4: 1x1 convolution followed by average pooling
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
    
    # Concatenate the outputs of all paths along the channel dimension
    concatenated_paths = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Add dropout to mitigate overfitting
    dropout = Dropout(0.5)(concatenated_paths)
    
    # Flatten the concatenated result
    flattened_paths = Flatten()(dropout)
    
    # Output layer
    dense1 = Dense(units=128, activation='relu')(flattened_paths)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model