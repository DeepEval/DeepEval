import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split input into three groups
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split1_1 = Lambda(lambda x: keras.backend.reshape(x, (-1, 1, 1, 3)))(split1[0])
    split1_2 = Lambda(lambda x: keras.backend.reshape(x, (-1, 3, 1, 1)))(split1[1])
    split1_3 = Lambda(lambda x: keras.backend.reshape(x, (-1, 3, 1, 3)))(split1[2])
    
    # Convolutional layers
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(split1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(split1_2)
    conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(split1_3)
    
    # Second block: Branch-wise feature extraction
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool = MaxPooling2D(pool_size=(1, 1))(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool)
    conv2_4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Concatenate and normalize outputs
    concat = Concatenate()([conv1_2, conv1_3, conv2_3, conv2_4])
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model