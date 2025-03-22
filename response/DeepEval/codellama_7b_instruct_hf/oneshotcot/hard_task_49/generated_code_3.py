import keras
from keras.layers import Input, Dense, Flatten, Concatenate, GlobalAveragePooling2D, Lambda, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1 = GlobalAveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    block1 = Flatten()(block1)
    block1 = Dense(units=128, activation='relu')(block1)
    block1 = Dense(units=64, activation='relu')(block1)
    
    # Block 2
    block2 = Reshape((4, 1, 1))(block1)
    block2 = Lambda(lambda x: tf.split(x, 4, axis=1))(block2)
    block2 = Concatenate(axis=1)(block2)
    block2 = Dense(units=128, activation='relu')(block2)
    block2 = Dense(units=64, activation='relu')(block2)
    block2 = Dense(units=10, activation='softmax')(block2)
    
    # Final model
    model = keras.Model(inputs=input_layer, outputs=block2)
    
    return model