import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=3))(input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split1[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split1[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split1[2])
    
    concat = Concatenate()([conv1, conv2, conv3])
    
    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine outputs
    fused = Add()([concat, branch])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(fused)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model