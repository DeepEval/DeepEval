import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same')(conv1)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        
        branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        output_tensor = Concatenate()([max_pool1, branch])
        
        return output_tensor

    def block2(input_tensor):
        split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=1))(input_tensor)
        split1[0] = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        split1[1] = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        split1[2] = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        output_tensor = Concatenate()(split1)
        
        output_tensor = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
        output_tensor = Flatten()(output_tensor)
        output_tensor = Dense(units=128, activation='relu')(output_tensor)
        output_layer = Dense(units=10, activation='softmax')(output_tensor)

        model = Model(inputs=input_layer, outputs=output_layer)
        
        return model

    block1_output = block1(input_layer)
    block2_output = block2(block1_output)

    model = Model(inputs=input_layer, outputs=block2_output)

    return model