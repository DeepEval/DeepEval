import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add average pooling layer with 1x1 window and stride
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(conv1)
    
    # Step 4: Add average pooling layer with 2x2 window and stride
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv1)
    
    # Step 5: Add average pooling layer with 4x4 window and stride
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(conv1)
    
    # Step 6: Flatten the outputs of the average pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Step 7: Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Step 8: Flatten the concatenated output
    flattened_concat = Flatten()(concatenated)
    
    # Step 9: Add dense layer
    dense1 = Dense(units=256, activation='relu')(flattened_concat)
    
    # Step 10: Add dense layer
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Step 11: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 12: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model