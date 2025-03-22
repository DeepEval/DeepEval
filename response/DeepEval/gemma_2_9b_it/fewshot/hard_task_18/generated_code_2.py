import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Second block
    main_path = GlobalAveragePooling2D()(avg_pool)
    
    fc1 = Dense(units=64, activation='relu')(main_path)
    fc2 = Dense(units=64, activation='relu')(fc1)
    refined_weights = Reshape(target_shape=(64, 1, 1))(fc2) 

    
    combined = Add()([input_layer, refined_weights * avg_pool]) 
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model