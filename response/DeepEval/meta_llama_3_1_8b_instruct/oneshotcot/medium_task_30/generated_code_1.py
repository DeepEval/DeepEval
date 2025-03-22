import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Average pooling layer with 1x1 pooling window and stride of 1x1
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv)
    
    # Average pooling layer with 2x2 pooling window and stride of 2x2
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
    
    # Average pooling layer with 4x4 pooling window and stride of 4x4
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv)
    
    # Concatenate the outputs of the three pooling layers
    output_tensor = Concatenate()([pool1, pool2, pool3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(output_tensor)
    
    # Dense layer with 128 units and ReLU activation
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Dense layer with 64 units and ReLU activation
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with 10 units and softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model