import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling3)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
    
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling4)
    max_pooling5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)
    
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling5)
    max_pooling6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)
    
    conv7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling6)
    max_pooling7 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv7)
    
    def block(input_tensor):

        conv1 = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling7)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model