import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split1[0] = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
    split1[1] = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
    split1[2] = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
    
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split2[0] = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
    split2[1] = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(split2[1])
    split2[2] = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(split2[2])
    split2[1] = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(split2[1])
    
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split3[0] = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split3[0])
    split3[1] = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split3[1])
    split3[2] = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split3[2])
    split3[1] = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(split3[1])
    split3[2] = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(split3[2])
    split3[1] = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(split3[1])
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()(split1 + split2 + split3)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()