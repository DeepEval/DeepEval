import keras
from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Reshape, Lambda

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)

        # Flatten and dropout each path
        path1 = Flatten()(path1)
        path2 = Flatten()(path2)
        path3 = Flatten()(path3)
        path1 = Dropout(0.2)(path1)
        path2 = Dropout(0.2)(path2)
        path3 = Dropout(0.2)(path3)

        # Concatenate the paths
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
    
    block1_output = block1(conv)
    dense_layer = Dense(units=128, activation='relu')(block1_output)
    reshaped_layer = Reshape((4,))(dense_layer)
    block2_input = Lambda(lambda x: x)(reshaped_layer)
    
    def block2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate the branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])

        return output_tensor
        
    block2_output = block2(Reshape((1, 1, 128))(block2_input))
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model