import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)
    
    # First block
    input_layer = Input(shape=input_shape)
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_1[2])
    
    concat_1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
    
    # Second block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_1)
    
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)
    conv2_4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_layer)
    concat_2 = Concatenate(axis=-1)([conv2_2, conv2_3, conv2_4])
    
    # Processing through both blocks
    processed_1 = BatchNormalization()(conv1_1)
    processed_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(processed_1)
    processed_2 = BatchNormalization()(conv2_1)
    processed_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(processed_2)
    
    # Flatten and dense layers
    flatten = Flatten()(processed_2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model