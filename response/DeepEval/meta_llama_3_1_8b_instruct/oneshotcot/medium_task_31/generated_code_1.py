import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    def split_image(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=2)
    
    split_layer = Lambda(split_image)(max_pooling)
    
    def conv_layer(input_tensor, filter_size):
        return Conv2D(filters=64, kernel_size=(filter_size, filter_size), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    path1 = conv_layer(split_layer[0], 1)
    path2 = conv_layer(split_layer[1], 3)
    path3 = conv_layer(split_layer[2], 5)
    
    output_tensor = Concatenate()([path1, path2, path3])
    
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()