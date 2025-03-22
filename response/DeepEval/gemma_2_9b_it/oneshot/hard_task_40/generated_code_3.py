import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    
    concat_layer = Concatenate()([flat1, flat2, flat3])
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    reshape_layer = Reshape((1, 128))(dense1) 

    # Second Block
    def block(input_tensor):
      
        conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_2)
        conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3_1)
        conv3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_2)
        avg_pool_conv = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        conv_avg_pool = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool_conv) 

        dropout_1x1 = Dropout(0.2)(conv1x1_1)
        dropout_3x3 = Dropout(0.2)(conv3x3_2)
        dropout_3x3_3 = Dropout(0.2)(conv3x3_3)
        dropout_avg_pool = Dropout(0.2)(conv_avg_pool)
        
        concat_output = Concatenate()([dropout_1x1, dropout_3x3, dropout_3x3_3, dropout_avg_pool])

        return concat_output
    
    block_output = block(reshape_layer)

    dense2 = Dense(units=64, activation='relu')(block_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model