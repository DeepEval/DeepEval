import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Reshape, AveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = BatchNormalization()(main_path)
        
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch = BatchNormalization()(branch)
        
        return main_path, branch

    def level1(input_tensor):
        main_path, branch = basic_block(input_tensor)
        adding_layer = Add()([main_path, branch])
        return adding_layer

    def level2(input_tensor):
        main_path, branch = basic_block(input_tensor)
        main_path, branch = basic_block(main_path)
        adding_layer = Add()([main_path, branch])
        return adding_layer

    def level3(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        return conv

    level1_output = level1(conv1)
    level2_output = level2(level1_output)
    level3_output = level3(level2_output)
    adding_layer = Add()([level3_output, level2_output])
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(adding_layer)
    flatten = Reshape(target_shape=(4*4*16,))(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model