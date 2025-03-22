import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():     
    input_layer = Input(shape=(32, 32, 3)) 
    
    # 1x1 Max Pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat1 = Flatten()(pool1)
    
    # 2x2 Max Pooling
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat2 = Flatten()(pool2)
    
    # 4x4 Max Pooling
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat3 = Flatten()(pool3)

    # Concatenate pooled features
    merged_features = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(merged_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model