import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv1)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_pool1)
    main_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv2)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_pool = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(branch_conv)
    
    # Combine paths
    combined = Add()([main_pool2, branch_pool])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model