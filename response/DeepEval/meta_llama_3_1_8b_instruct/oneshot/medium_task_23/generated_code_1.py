import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2
    path2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_conv1)
    path2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_conv2)
    
    # Path 3
    path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_conv2_1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_conv1)
    path3_conv2_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_conv2_1)
    path3_conv3_1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_conv2_2)
    path3_conv3_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_conv3_1)
    path3_conv4 = Concatenate()([path3_conv2_2, path3_conv3_1, path3_conv3_2])
    
    # Path 4
    path4_avg_pool = AveragePooling2D(pool_size=(8, 8), strides=8, padding='same')(input_layer)
    path4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4_avg_pool)
    
    # Concatenate the outputs
    output_tensor = Concatenate()([path1_conv, path2_conv3, path3_conv4, path4_conv])
    
    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(batch_norm)
    
    # Apply fully connected layers for classification
    dense = Dense(units=384, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model