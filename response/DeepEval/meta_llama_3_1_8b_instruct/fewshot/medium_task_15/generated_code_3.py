import keras
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)
    
    global_avg_pool = GlobalAveragePooling2D()(batch_norm)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape the output to match the size of the initial feature
    reshaped = Reshape(target_shape=(32, 32, 64))(dense2)
    
    # Multiply the reshaped output with the initial feature
    weighted_feature = Multiply()([conv, reshaped])
    
    # Concatenate the weighted feature with the input layer
    concat = Concatenate()([input_layer, weighted_feature])
    
    # Reduce dimensionality and downsample the feature using 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)
    
    # Flatten the output and produce the final classification result
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model