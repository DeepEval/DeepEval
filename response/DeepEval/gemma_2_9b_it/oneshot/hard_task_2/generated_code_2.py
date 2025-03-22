import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Split input into three channels
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
        
        # Process each channel group
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)

        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)

        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)

        # Concatenate group outputs
        concat_output = Add()([group1, group2, group3])

        # Fuse with original input
        final_output = Add()([concat_output, input_layer])

        # Flatten and classify
        flatten_layer = Flatten()(final_output)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model