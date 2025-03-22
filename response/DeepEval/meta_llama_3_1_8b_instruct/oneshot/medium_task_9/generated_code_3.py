import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.layers import AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)

    def basic_block(input_tensor):

        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        
        output_tensor = Add()([input_tensor, batch_norm1])
        return output_tensor
        
    block1_output = basic_block(batch_norm)
    block2_output = basic_block(block1_output)

    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    branch_add = Add()([block2_output, branch_conv])

    fusion = Add()([block2_output, branch_add])
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(fusion)
    
    avg_pooling = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(max_pooling)
    flatten_layer = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model