import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define main path
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    
    # Define branch path
    block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3)
    
    # Combine main and branch paths using addition
    combined_path = Add()([max_pooling1, max_pooling3])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(combined_path)
    flatten_layer = Flatten()(bath_norm)
    
    # Define two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Define model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model