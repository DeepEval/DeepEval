import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: <max pooling>
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Add batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)
    
    # Add dropout layers to mitigate overfitting
    dropout1 = Dropout(0.5)(flatten_layer)
    dropout2 = Dropout(0.5)(dropout1)
    
    # Add dense layers for classification
    dense1 = Dense(units=256, activation='relu')(dropout2)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model