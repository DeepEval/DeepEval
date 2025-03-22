import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Reshape, GlobalAveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Second block
    main_path = GlobalAveragePooling2D()(avg_pool)
    main_path = Dense(units=64, activation='relu')(main_path) 
    main_path = Dense(units=64, activation='relu')(main_path)
    
    refined_weights = Reshape((1, 1, 64))(main_path) # Reshape to match input dimensions
    
    # Combine paths
    output = Add()([avg_pool, refined_weights * input_layer])
    
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model