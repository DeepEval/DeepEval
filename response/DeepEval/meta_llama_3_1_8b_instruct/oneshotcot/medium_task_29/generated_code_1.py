import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add first maxpooling layer with window size 1x1
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Step 3: Add second maxpooling layer with window size 2x2
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(max_pooling_1)
    
    # Step 4: Add third maxpooling layer with window size 4x4
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(max_pooling_2)
    
    # Step 5: Flatten each max pooling layer into a one-dimensional vector
    flatten_1 = Flatten()(max_pooling_1)
    flatten_2 = Flatten()(max_pooling_2)
    flatten_3 = Flatten()(max_pooling_3)
    
    # Step 6: Concatenate the flattened vectors to form a unified feature set
    concat_features = Concatenate()([flatten_1, flatten_2, flatten_3])
    
    # Step 7: Add the first dense layer
    dense1 = Dense(units=128, activation='relu')(concat_features)
    
    # Step 8: Add the second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 10: Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model