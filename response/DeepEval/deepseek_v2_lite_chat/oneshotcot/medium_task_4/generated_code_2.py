import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, Concatenate

def dl_model():
    # Path 1: Two blocks of convolution, average pooling
    path1_input = Input(shape=(32, 32, 3), name="Path1_Input")
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path1_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    avg_pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Path2: Single convolutional layer
    path2_input = Input(shape=(32, 32, 3), name="Path2_Input")
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(path2_input)
    
    # Combine Path1 and Path2
    merged = Concatenate()([avg_pool1, conv3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(merged)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=[path1_input, path2_input], outputs=output)
    
    return model