import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    def block1():
        input_layer = Input(shape=(28, 28, 1))
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        avg_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(conv3)
        avg_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        avg_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv3)
        
        flat1 = Flatten()(avg_pool3)
        flat2 = Flatten()(Concatenate()([avg_pool1, avg_pool2, flat1]))
        
        return keras.Model(inputs=input_layer, outputs=[flat1, flat2])

    def block2():
        input_tensor = Input(shape=(None,))
        input_tensor = Dense(units=64)(input_tensor)
        
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path6 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Concatenate()([path1, path2, path3, path4, path5, path6])

        return output_tensor

    flat1, flat2 = block1()
    output_tensor = block2()(flat2)

    dense1 = Dense(units=128, activation='relu')(output_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])