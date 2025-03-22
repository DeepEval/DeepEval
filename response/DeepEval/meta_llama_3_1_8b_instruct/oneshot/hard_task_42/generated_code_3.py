import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    # Concatenate the pooling results along the channel dimension
    block1_output = Concatenate()([path1, path2, path3])
    
    # Flatten the output of Block 1 and apply dropout regularization
    flatten_layer = Flatten()(block1_output)
    dropout1 = Dropout(0.2)(flatten_layer)
    
    # Reshape the output of the fully connected layer to a 4-dimensional tensor
    reshaped_layer = Reshape((4, 4, 128))(dropout1)
    
    # Block 2
    path1_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_layer)
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_layer)
    path2_block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_block2)
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_layer)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path4_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(reshaped_layer))
    
    # Concatenate the outputs of all paths along the channel dimension
    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])
    
    # Apply batch normalization and flatten the output
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Two fully connected layers to produce the final classification results
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout2 = Dropout(0.2)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout2)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model