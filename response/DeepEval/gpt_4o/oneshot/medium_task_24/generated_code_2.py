import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution followed by 3x3 convolution
    branch1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_1x1)
    branch1_dropout = Dropout(0.3)(branch1_3x3)
    
    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_1x7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_1x1)
    branch2_7x1 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_1x7)
    branch2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_7x1)
    branch2_dropout = Dropout(0.3)(branch2_3x3)
    
    # Branch 3: Max pooling
    branch3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3_dropout = Dropout(0.3)(branch3_pool)
    
    # Concatenate all branches
    concatenated = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])
    
    # Flatten and add fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model