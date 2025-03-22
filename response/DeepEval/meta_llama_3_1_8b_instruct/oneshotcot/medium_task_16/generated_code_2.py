import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, Flatten, Dense
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_input(x):
        return tf.split(x, 3, axis=-1)
    
    split_input_layer = Lambda(split_input)(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    group1 = Conv2D(int(32/3), (1, 1), activation='relu')(split_input_layer[0])
    group2 = Conv2D(int(32/3), (1, 1), activation='relu')(split_input_layer[1])
    group3 = Conv2D(int(32/3), (1, 1), activation='relu')(split_input_layer[2])
    
    # Apply average pooling to each group
    group1 = AveragePooling2D(pool_size=(2, 2), strides=2)(group1)
    group2 = AveragePooling2D(pool_size=(2, 2), strides=2)(group2)
    group3 = AveragePooling2D(pool_size=(2, 2), strides=2)(group3)
    
    # Concatenate the three groups along the channel dimension
    concatenated_layer = Concatenate()([group1, group2, group3])
    
    # Flatten the concatenated feature maps
    flattened_layer = Flatten()(concatenated_layer)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model