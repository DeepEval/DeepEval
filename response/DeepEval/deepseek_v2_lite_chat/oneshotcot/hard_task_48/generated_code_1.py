import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, [1, 1, 1], axis=2))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split1[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split1[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split1[2])
        concat1 = Concatenate()([conv1, conv2, conv3])
        return concat1
    
    def block2(input_tensor):
        avg_pool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(avg_pool)
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
        sub_path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv5)
        sub_path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
        sub_path3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(conv5)
        sub_path4 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(conv5)
        concat2 = Concatenate()([sub_path1, sub_path2, sub_path3, sub_path4])
        return concat2
    
    input_layer = Input(shape=(32, 32, 3))
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model