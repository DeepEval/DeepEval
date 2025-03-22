import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, Flatten, Dense
from keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(batch_norm1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(relu1)
        batch_norm2 = BatchNormalization()(conv2)
        relu2 = Activation('relu')(batch_norm2)
        
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(relu2)
        batch_norm3 = BatchNormalization()(conv3)
        relu3 = Activation('relu')(batch_norm3)
        
        output_tensor = Add()([conv1, conv2, conv3])
        
        return output_tensor
    
    block1_output = block(conv)
    block2_output = block(block1_output)
    block3_output = block(block2_output)
    added_output = Add()([block1_output, block2_output, block3_output])
    
    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model