import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split input into three groups for multi-scale feature extraction
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Main path features extraction
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
    
    # Branch path features extraction
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
    
    # Concatenate and add the main path outputs
    concat = Concatenate()(
        [conv1x1, conv3x3, conv5x5, conv1]
    )
    add = Add()([concat, inputs])
    
    # Batch normalization, flattening, and dense layers
    bn = BatchNormalization()(add)
    flat = Flatten()(bn)
    
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    outputs = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model