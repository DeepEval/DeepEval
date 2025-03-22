import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dropout, BatchNormalization, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    conv1a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1b = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1c = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    pool1_flatten = Flatten()(pool1)
    pool1_dropout = Dropout(0.5)(pool1_flatten)
    
    concat1 = Concatenate()(pool1_dropout)
    
    # Block 2
    conv2a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv2b = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv2c = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat1)
    conv2d = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat1)
    avg_pool = MaxPooling2D(pool_size=(1, 7), strides=(1, 1), padding='same')(concat1)
    pool2_flatten = Flatten()(avg_pool)
    pool2_dropout = Dropout(0.5)(pool2_flatten)
    
    concat2 = Concatenate(axis=-1)(pool2_dropout)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(concat2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model