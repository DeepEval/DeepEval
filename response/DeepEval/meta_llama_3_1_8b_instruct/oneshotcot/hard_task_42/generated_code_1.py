import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    path1_flat = Flatten()(path1)
    path2_flat = Flatten()(path2)
    path3_flat = Flatten()(path3)
    
    path1_flat = Dense(128, activation='relu')(path1_flat)
    path2_flat = Dense(128, activation='relu')(path2_flat)
    path3_flat = Dense(128, activation='relu')(path3_flat)
    
    path1_flat = Dropout(0.2)(path1_flat)
    path2_flat = Dropout(0.2)(path2_flat)
    path3_flat = Dropout(0.2)(path3_flat)
    
    block1_output = Concatenate()([path1_flat, path2_flat, path3_flat])
    
    # Transformation layer for block 2
    transform_layer = Dense(256, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 64))(transform_layer)
    
    # Block 2
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path4)
    
    output_tensor = Concatenate()([path1, path2, path3, path4])
    
    # Flatten layer and fully connected layers
    output_tensor = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model