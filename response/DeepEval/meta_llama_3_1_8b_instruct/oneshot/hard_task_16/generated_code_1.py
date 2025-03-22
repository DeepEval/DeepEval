import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras import regularizers
from tensorflow.keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[1])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
        group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[2])
        group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
        group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)
        output_tensor = Concatenate()([group1, group2, group3])
        
        return output_tensor
    
    block_output = block1(input_layer)
    transition_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_output)
    
    def block2(input_tensor):
        channel_weights = Dense(units=96, activation='relu')(input_tensor)
        channel_weights = Dense(units=96)(channel_weights)
        channel_weights = Reshape((1, 1, 96))(channel_weights)
        output_tensor = Multiply()([input_tensor, channel_weights])
        output_tensor = GlobalAveragePooling2D()(output_tensor)
        
        return output_tensor
    
    block_output = block2(transition_conv)
    main_path = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(block_output)
    
    branch = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    output = Add()([main_path, branch])
    
    output_layer = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model