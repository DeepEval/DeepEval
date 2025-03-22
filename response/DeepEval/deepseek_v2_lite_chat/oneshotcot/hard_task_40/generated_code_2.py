import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: process through three average pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    # Flatten and reshape for the second block
    flat = Flatten()(Concatenate()([pool1, pool2, pool3]))
    reshape = Reshape((-1, reshape.shape[1]*reshape.shape[2]*reshape.shape[3]))(flat)
    
    # Second block: four parallel paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape)
    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape)
    
    # Dropout for each path to mitigate overfitting
    path1 = Dropout(0.5)(path1)
    path2 = Dropout(0.5)(path2)
    path3 = Dropout(0.5)(path3)
    path4 = Dropout(0.5)(path4)
    
    # Concatenate the outputs from all paths
    concat = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model