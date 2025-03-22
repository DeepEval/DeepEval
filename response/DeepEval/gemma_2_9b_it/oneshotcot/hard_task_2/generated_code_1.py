import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def split_and_process(input_tensor):
        split_tensor = tf.split(input_tensor, num_or_size_splits=3, axis=2)  
        
        # Process each group
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        
        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
        
        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)

        # Concatenate the processed groups
        combined = tf.concat([group1, group2, group3], axis=2) 

        return combined

    processed_input = split_and_process(input_layer)
    
    # Fuse with original input
    fused_features = Add()([processed_input, input_layer])
    
    # Flatten and classify
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model