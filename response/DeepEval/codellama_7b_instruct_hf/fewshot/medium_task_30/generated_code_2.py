import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    fused_features = Concatenate()([flatten1, flatten2, flatten3])
    fused_features = Flatten()(fused_features)
    fused_features = Dense(units=128, activation='relu')(fused_features)
    fused_features = Dense(units=64, activation='relu')(fused_features)
    output_layer = Dense(units=10, activation='softmax')(fused_features)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model