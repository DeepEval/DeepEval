import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    lambda_layer = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1_1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(lambda_layer[0])
    conv1_3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1_1)
    conv1_1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_3_1)
    
    conv2_1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(lambda_layer[1])
    conv2_3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1_1)
    conv2_1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_3_1)
    
    conv3_1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(lambda_layer[2])
    conv3_3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1_1)
    conv3_1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3_3_1)
    
    main_path = Add()([conv1_1_2, conv2_1_2, conv3_1_2])
    
    model = keras.Model(inputs=input_layer, outputs=main_path)

    return model