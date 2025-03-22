import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Splitting input into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path1 = BatchNormalization()(path1)

        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path2 = BatchNormalization()(path2)

        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        path3 = BatchNormalization()(path3)

        # Concatenate the outputs of the three paths
        block1_output = Concatenate()([path1, path2, path3])
        return block1_output

    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        path2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)

        path3_input = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_a = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_input)
        path3_b = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_input)
        path3 = Concatenate()([path3_a, path3_b])

        path4_input = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_a = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_input)
        path4_b = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_input)
        path4 = Concatenate()([path4_a, path4_b])
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)

        # Concatenate the outputs of the four paths
        block2_output = Concatenate()([path1, path2, path3, path4])
        return block2_output

    block2_output = block2(block1_output)

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model