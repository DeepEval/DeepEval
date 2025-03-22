import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Add, AveragePooling2D, Concatenate, BatchNormalization

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Add batch normalization layer
    batch_norm = BatchNormalization()(conv)
    
    # Step 3: Add global average pooling layer
    global_pool = GlobalAveragePooling2D()(batch_norm)
    
    # Step 4: Add first fully connected layer
    dense1 = Dense(units=128, activation='relu')(global_pool)
    
    # Step 5: Add second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 6: Reshape to match the size of the initial feature
    reshape = Reshape((32, 32, 64))(dense2)
    
    # Step 7: Multiply with the initial feature
    multiply = Multiply()([reshape, conv])
    
    # Step 8: Add concatenated input layer
    concat = Concatenate()([multiply, input_layer])
    
    # Step 9: Add 1x1 convolution layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Step 10: Add average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=8, padding='same')(conv1x1)
    
    # Step 11: Add final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(avg_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model