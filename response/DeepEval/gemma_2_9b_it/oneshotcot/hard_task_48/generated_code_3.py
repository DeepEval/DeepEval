import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Block 1
        x = Lambda(lambda tensor: tf.split(tensor, 3, axis=3))(input_layer)
        
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
        branch1 = BatchNormalization()(branch1)

        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
        branch2 = BatchNormalization()(branch2)

        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
        branch3 = BatchNormalization()(branch3)

        x = Concatenate()([branch1, branch2, branch3])

        # Block 2
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path1 = BatchNormalization()(x)

        x = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path2 = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x)
        path3_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path3 = Concatenate()([path3_1, path3_2])
        path3 = BatchNormalization()(path3)

        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path4_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(x)
        path4_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(x)
        path4 = Concatenate()([path4_1, path4_2])
        path4 = BatchNormalization()(path4)

        x = Concatenate()([path1, path2, path3, path4])

        x = Flatten()(x)
        output_layer = Dense(units=10, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model