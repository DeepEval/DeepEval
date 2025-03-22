import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Lambda, tf
from keras.layers import Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Block 1
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path) 

        branch_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        block1_output = Add()([main_path, branch_path])

        # Block 2
        x = Lambda(lambda x: tf.split(x, 3, axis=2))(block1_output)
        
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
        
        block2_output = Concatenate()([path1, path2, path3])
        
        # Fully connected layers
        flatten_layer = Flatten()(block2_output)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model