import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, Concatenate, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Three Average Pooling Layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the outputs and concatenate
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    concat_flat = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer followed by a reshape operation
    dense_fc = Dense(units=128, activation='relu')(concat_flat)
    reshape_tensor = Reshape((4, 4, 8))(dense_fc)  # Reshape to a 4D tensor
    
    # Second Block: Four Parallel Paths
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_tensor)
    path1 = Dropout(rate=0.5)(path1)
    
    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_tensor)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Dropout(rate=0.5)(path2)
    
    # Path 3: 1x1 Convolution followed by a 3x3 Convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_tensor)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Dropout(rate=0.5)(path3)
    
    # Path 4: Average Pooling followed by a 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshape_tensor)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Dropout(rate=0.5)(path4)
    
    # Concatenate the outputs from all paths
    concat_paths = Concatenate()([path1, path2, path3, path4])
    
    # Final Batch Normalization
    batch_norm = BatchNormalization()(concat_paths)
    
    # Flatten and pass through dense layers for classification
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model