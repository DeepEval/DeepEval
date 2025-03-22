import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1x1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Branch path
    conv4 = Conv2D(filters=64, kernel_size=(1x1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Concatenate the outputs of the two paths
    concat_layer = Concatenate(axis=-1)([conv3, conv5])
    
    # Addition
    add_layer = Add()([conv3, conv5])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model