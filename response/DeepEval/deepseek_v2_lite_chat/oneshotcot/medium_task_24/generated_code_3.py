import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.5)(branch1)
    
    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.5)(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    branch3 = Dropout(rate=0.5)(branch3)
    
    # Concatenate the outputs of all branches
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model