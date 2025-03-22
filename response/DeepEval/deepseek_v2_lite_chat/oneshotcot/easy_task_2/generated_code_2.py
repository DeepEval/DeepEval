import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, Dropout

def dl_model():
    input_layer = Input(shape=(224, 224, 3))

    # Feature extraction layers
    fe1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    fe1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(fe1)
    fe2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(fe1)
    fe2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(fe2)

    # Convolutional layers for the block
    cb1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(fe2)
    cb1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(cb1)
    cb2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(cb1)
    cb2 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu')(cb2)

    # Concatenate and average pool the features
    concat = Concatenate()(cb2)
    avg_pool = MaxPooling2D(pool_size=(7, 7), strides=1, padding='valid')(concat)

    # Flatten and pass through dense layers
    flat = Flatten()(avg_pool)
    dense1 = Dense(units=256, activation='relu')(flat)
    drop1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(drop1)
    drop2 = Dropout(rate=0.5)(dense2)
    output_layer = Dense(units=1000, activation='softmax')(drop2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model