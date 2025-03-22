import keras
from keras.layers import Input, DepthwiseConv2D, Lambda, Add, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def channel_wise_feature_transformation(input_tensor):
        dense1 = Dense(units=128, activation='relu')(input_tensor)
        dense2 = Dense(units=128, activation='relu')(dense1)
        return dense2
    
    x = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Lambda(lambda x: keras.layers.LayerNormalization(axis=-1)(x))(x)
    
    output1 = channel_wise_feature_transformation(x)
    
    x = Add()([input_layer, output1])
    
    output2 = channel_wise_feature_transformation(x)
    
    x = Add()([input_layer, output2])
    
    flatten_layer = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model