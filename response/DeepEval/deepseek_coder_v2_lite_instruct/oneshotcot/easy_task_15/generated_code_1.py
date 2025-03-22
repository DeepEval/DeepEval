import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_1)
    dropout1 = Dropout(0.25)(avg_pool1)
    
    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_1)
    dropout2 = Dropout(0.25)(avg_pool2)
    
    # Global average pooling
    gap = keras.layers.GlobalAveragePooling2D()(dropout2)
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(gap)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model