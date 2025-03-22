import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Step 3: Flatten the outputs of the pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Step 4: Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Step 5: Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 6: Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model