import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K
from tensorflow.keras import layers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def separable_convolution(kernel_size):
        return layers.SeparableConv2D(
            filters=64,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            activation='relu'
        )
        
    main_path = Lambda(lambda x: K.tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path = Lambda(lambda x: [separable_convolution(1)(i) for i in x] + 
                       [separable_convolution(3)(i) for i in x] + 
                       [separable_convolution(5)(i) for i in x])(main_path)
    main_path = Concatenate()(main_path)
    
    # Branch path
    branch_path = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(input_layer)
    
    # Fusion
    output = layers.Add()([main_path, branch_path])
    
    # Flatten and fully connected layers
    bath_norm = BatchNormalization()(output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model