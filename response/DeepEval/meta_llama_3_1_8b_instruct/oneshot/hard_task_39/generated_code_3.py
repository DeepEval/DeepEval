import keras
from keras.layers import Input, MaxPooling2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool1_flat = Flatten()(pool1)
        
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool2_flat = Flatten()(pool2)
        
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        pool3_flat = Flatten()(pool3)
        
        output_tensor = Concatenate()([pool1_flat, pool2_flat, pool3_flat])
        return output_tensor

    block1_output = block1(input_layer)
    
    # Fully connected layer and reshape operation
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape(target_shape=(2, 64))(dense1)
    
    # Block 2
    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    block2_output = block2(reshape_layer)
    
    # Flatten and fully connected layer for classification
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model