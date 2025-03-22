import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
        batch_norm3 = BatchNormalization()(conv3)
        output_tensor = conv + conv1 + conv2 + conv3  # Add the outputs to enhance feature representation

        return output_tensor
        
    block_output = block(conv)
    batch_norm = BatchNormalization()(block_output)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm)
    block_output2 = block(max_pooling)  # Three parallel blocks
    batch_norm2 = BatchNormalization()(block_output2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm2)
    block_output3 = block(max_pooling2)  
    batch_norm3 = BatchNormalization()(block_output3)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm3)
    flatten_layer = Flatten()(max_pooling3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model