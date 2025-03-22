import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv_layer = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def basic_block(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn)
        output_tensor = Add()([conv1, conv2])
        return output_tensor
    
    block1 = basic_block(input_tensor=conv_layer)
    
    def residual_block(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn2)
        output_tensor = Add()([conv1, conv2, conv3])
        return output_tensor
    
    block2 = residual_block(input_tensor=block1)
    block3 = residual_block(input_tensor=block2)
    
    def global_branch(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv)
        return output_tensor
    
    global_branch_output = global_branch(input_tensor=conv_layer)
    
    bn3 = BatchNormalization()(block3)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(bn3)
    
    output = Add()([maxpool, global_branch_output])
    
    flatten_layer = Flatten()(output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model