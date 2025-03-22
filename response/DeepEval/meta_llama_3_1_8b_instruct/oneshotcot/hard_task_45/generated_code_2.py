import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import regularizers
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    block1_output = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output[0])
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output[1])
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_output[2])
    
    # Concatenate the outputs from the above paths
    output1 = Concatenate()([path1, path2, path3])
    
    # Second Block
    block2_output = layers.Lambda(lambda x: tf.split(x, 6, axis=-1))(output1)
    
    # Branch 1
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output[0])
    
    # Branch 2
    path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output[1])
    path6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output[1])
    
    # Branch 3
    path7 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(block2_output[2])
    path8 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path7)
    
    # Branch 4
    path9 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output[3])
    path10 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output[3])
    path11 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2_output[3])
    
    # Branch 5
    path12 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output[4])
    path13 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output[4])
    
    # Branch 6
    path14 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output[5])
    path15 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output[5])
    path16 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block2_output[5])
    
    # Concatenate the outputs from all branches
    output2 = Concatenate()([path4, path5, path6, path8, path9, path10, path11, path12, path13, path14, path15, path16])
    
    # Batch Normalization
    bath_norm = BatchNormalization()(output2)
    
    # Flatten the result
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model