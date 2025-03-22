import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu1)
    
    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu2)
    
    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu3)
    
    # Parallel Branch
    conv_branch = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate Outputs
    concat = Concatenate()([max_pool3, conv_branch])
    
    # Fully Connected Layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model