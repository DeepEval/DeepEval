import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction at multiple scales
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(4, 4), padding='same')(input_layer)
    
    # Flatten and concatenate features
    concat = Concatenate()(
        [Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool1),
         Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2),
         Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(avg_pool3)])
    
    # Output layer
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model