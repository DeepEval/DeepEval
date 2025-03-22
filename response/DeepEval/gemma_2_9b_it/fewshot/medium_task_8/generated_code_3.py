import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        group1 = split_tensor[0]
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group3 = split_tensor[2]
        
        combined_group2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        
        combined = Concatenate()([group1, combined_group2_3, group3])

        return combined
    
    def branch_path(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    added_output = keras.layers.Add()([main_output, branch_output])
    
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model