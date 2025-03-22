import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    def group_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    split_input_layer = split_input(input_layer)
    
    group1 = group_block(split_input_layer[0])
    group2 = group_block(split_input_layer[1])
    group3 = group_block(split_input_layer[2])
    
    main_path = Add()([group1, group2, group3])
    
    combined_path = Add()([main_path, input_layer])
    
    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model