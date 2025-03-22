import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Pooling layers
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(maxpool1)
    
    # Paths in the block
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(maxpool2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool2)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(maxpool2)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=2)(maxpool2)
    
    # Concatenate paths
    concat_layer = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model