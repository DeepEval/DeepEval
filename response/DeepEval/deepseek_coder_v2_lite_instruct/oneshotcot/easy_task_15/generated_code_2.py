import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First specialized block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    block1_dropout = Dropout(0.25)(block1_avg_pool)
    
    # Second specialized block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_dropout)
    block2_1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2_dropout = Dropout(0.25)(block2_avg_pool)
    
    # Global average pooling and flattening
    global_avg_pool = keras.layers.GlobalAveragePooling2D()(block2_dropout)
    flatten_layer = Flatten()(global_avg_pool)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model