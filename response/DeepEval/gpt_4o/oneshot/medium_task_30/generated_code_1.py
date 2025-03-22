import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Three average pooling layers with different pooling sizes
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the outputs of the pooling layers
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concatenated)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model