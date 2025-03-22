import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction for the first group
    fe1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[0])
    
    # Feature extraction for the second group
    fe2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
    
    # Feature extraction for the third group
    fe3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[0])
    
    # Concatenate the outputs from the three groups
    concat = Concatenate(axis=-1)([fe1, fe2, fe3])
    
    # Additional 3x3 convolution
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # MaxPooling2D layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv3x3)
    
    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate the outputs from the main and branch paths
    fused = keras.backend.concatenate([pool, branch])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(fused)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model