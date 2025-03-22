import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def feature_extraction(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    channel_groups = tf.split(input_layer, num_or_size_splits=3, axis=-1)
    extracted_features = []
    for channel in channel_groups:
        extracted_features.append(feature_extraction(channel))

    concat_features = Concatenate(axis=-1)(extracted_features)
    bath_norm = BatchNormalization()(concat_features)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model