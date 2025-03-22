import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.layers import concatenate

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max Pooling with different scales
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    path1 = Flatten()(path1)
    
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2 = Flatten()(path2)
    
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    path3 = Flatten()(path3)
    
    block_output1 = concatenate([path1, path2, path3])
    
    # Fully connected layer and reshape operation
    dense_layer = Dense(units=128, activation='relu')(block_output1)
    reshape_layer = Reshape((4, 1))(dense_layer)
    
    # Block 2: Convolution and Max Pooling with different sizes
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path7 = MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(reshape_layer)
    output_tensor = concatenate([path4, path5, path6, path7])
    
    # Batch normalization, flatten and dense layer
    batch_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(batch_norm)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model