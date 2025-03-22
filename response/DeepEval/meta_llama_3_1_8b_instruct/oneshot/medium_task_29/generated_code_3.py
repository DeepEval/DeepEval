import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Apply three max pooling layers with varying window sizes
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(max_pooling1)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(max_pooling2)

    # Flatten the output from each pooling layer
    flatten1 = Flatten()(max_pooling1)
    flatten2 = Flatten()(max_pooling2)
    flatten3 = Flatten()(max_pooling3)

    # Concatenate the flattened vectors to form a unified feature set
    output_tensor = Concatenate()([flatten1, flatten2, flatten3])

    # Apply two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model