import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    split_lambda = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))([input_layer])
    
    # Splitting the input into three groups
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split_lambda[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(split_lambda[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(split_lambda[2])
    
    # Concatenate the outputs of the main path
    concat = Concatenate()(conv1, conv2, conv3)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model