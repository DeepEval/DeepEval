import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # First Block
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        # Second Block
        main_path = GlobalAveragePooling2D()(x)
        main_path = Dense(units=32, activation='relu')(main_path)
        main_path = Dense(units=32, activation='relu')(main_path)
        main_path = Reshape((32, 1, 1))(main_path)

        output = Add()([x, main_path * x])
        output = Flatten()(output)
        output = Dense(units=10, activation='softmax')(output)

        model = keras.Model(inputs=input_layer, outputs=output)

        return model