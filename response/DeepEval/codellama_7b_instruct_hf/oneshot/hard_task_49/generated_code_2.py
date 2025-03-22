import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(pool1)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(pool2)
    pool_out = Flatten()(pool3)
    
    # Second block
    first_block_output = pool_out
    first_block_output = Lambda(lambda x: tf.split(x, 4, axis=-1))(first_block_output)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_block_output[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block_output[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(first_block_output[2])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(first_block_output[3])
    second_block_output = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Fully connected layers
    second_block_output = Flatten()(second_block_output)
    second_block_output = Dense(units=128, activation='relu')(second_block_output)
    second_block_output = Dense(units=64, activation='relu')(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(second_block_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model