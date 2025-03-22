import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Direct input convolutional layer
    direct_input = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Convolutional paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(direct_input)
    
    # Concatenate paths
    concatenated = Concatenate()([path1, path2])
    
    # Pooling layer
    pooled = MaxPooling2D(pool_size=(2, 2))(concatenated)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(pooled)
    flatten = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model