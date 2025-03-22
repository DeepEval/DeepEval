import keras
from keras.layers import Input, Concatenate, Lambda, SeparableConv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def split_channels(input_tensor):
        channel_groups = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        return channel_groups
    
    split_layer = Lambda(split_channels)(input_layer)
    
    conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(split_layer[0])
    conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(split_layer[1])
    conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(split_layer[2])
    
    concat_1 = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    def branch_1(input_tensor):
        conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(input_tensor)
        return conv3x3
    
    branch_2 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(concat_1)
    branch_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_2)
    branch_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_2)
    
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat_1)
    
    concat_2 = Concatenate()([branch_2, maxpool])
    
    batch_norm = BatchNormalization()(concat_2)
    gavg_pool = GlobalAveragePooling2D()(batch_norm)
    dense_layer = Dense(units=64, activation='relu')(gavg_pool)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model