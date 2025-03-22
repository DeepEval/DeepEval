import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Left branch
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_3x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_3x3)
    
    # Right branch
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_5x5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_5x5)

    # Combine branches
    concat = Concatenate()([max_pooling_3x3, max_pooling_5x5])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(concat)
    
    # Fully connected layers
    dense_fc1 = Dense(units=128, activation='relu')(avg_pool)
    dense_fc2 = Dense(units=64, activation='relu')(dense_fc1)
    output_layer = Dense(units=10, activation='softmax')(dense_fc2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model