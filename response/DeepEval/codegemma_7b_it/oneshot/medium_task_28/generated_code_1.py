import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, GlobalAveragePooling2D, Reshape

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))
    attention_layer = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(input_layer)
    weighted_input = Multiply()([input_layer, attention_layer])
    
    contextual_info = GlobalAveragePooling2D()(weighted_input)
    contextual_info = Reshape((1, 1, -1))(contextual_info)
    contextual_info = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(contextual_info)
    
    input_with_attention = Concatenate()([input_layer, contextual_info])
    input_reduced = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_with_attention)
    input_reduced = BatchNormalization()(input_reduced)
    
    input_restored = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_reduced)
    input_with_attention = Concatenate()([input_restored, input_with_attention])
    
    flattened_layer = Flatten()(input_with_attention)
    dense1 = Dense(units=128, activation='relu')(flattened_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model