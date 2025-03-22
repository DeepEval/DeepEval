import keras
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    split_output = split_input(input_layer)
    
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    
    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
    
    group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
    group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)
    
    main_path = Add()([group1, group2, group3])
    combined_path = Add()([main_path, input_layer])
    flat_features = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flat_features)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model