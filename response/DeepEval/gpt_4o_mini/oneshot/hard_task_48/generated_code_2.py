import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    
    def block1_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        
        return BatchNormalization()(Concatenate()([conv1, conv3, conv5]))

    block1_outputs = [block1_group(tensor) for tensor in split_tensors]
    block1_concat = Concatenate()(block1_outputs)

    # Block 2
    def block2_path1(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

    def block2_path2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        return Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(avg_pool)

    def block2_path3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(conv1)
        conv3_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(conv1)
        return Concatenate()([conv3_1, conv3_2])

    def block2_path4(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(conv3)
        conv3_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(conv3)
        return Concatenate()([conv3_1, conv3_2])

    path1_output = block2_path1(block1_concat)
    path2_output = block2_path2(block1_concat)
    path3_output = block2_path3(block1_concat)
    path4_output = block2_path4(block1_concat)

    block2_concat = Concatenate()([path1_output, path2_output, path3_output, path4_output])

    # Final layers
    flatten_layer = Flatten()(block2_concat)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model