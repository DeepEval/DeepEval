import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras import layers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(input_tensor):
        split = tf.split(input_tensor, 3, axis=-1)
        conv1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
        conv3 = layers.SeparableConv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
        concat = Concatenate()([conv1, conv2, conv3])
        return concat
    block1_output = block1(input_layer)

    # Second block
    def block2(input_tensor):
        branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch4 = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(input_tensor)
        branch4 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        concat = Concatenate()([branch1, branch2, branch3, branch4])
        return concat
    block2_output = block2(block1_output)

    # Feature fusion
    flatten = Flatten()(block2_output)
    dense = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model