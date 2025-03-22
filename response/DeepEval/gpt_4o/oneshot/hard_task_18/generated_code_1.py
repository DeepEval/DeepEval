import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape, Flatten, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Combine input with main path output by addition
    addition_output = Add()([input_layer, avg_pooling])
    
    # Second Block
    # Global average pooling for channel-wise feature compression
    global_avg_pooling = GlobalAveragePooling2D()(addition_output)
    
    # Fully connected layers for channel weight refinement
    dense1 = Dense(units=64, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=addition_output.shape[-1], activation='sigmoid')(dense1)
    
    # Reshape to match the dimensions of addition_output for channel-wise scaling
    scale_weights = Reshape((1, 1, addition_output.shape[-1]))(dense2)
    
    # Multiply the input by the channel weights
    scaled_output = Multiply()([addition_output, scale_weights])
    
    # Flatten and pass through a final fully connected layer for classification
    flatten_layer = Flatten()(scaled_output)
    batch_norm = BatchNormalization()(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(batch_norm)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model