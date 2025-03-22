import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))


    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path, main_path])


    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)


    output = main_path + branch_path


    output = Flatten()(output)


    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)


    model = keras.Model(inputs=input_layer, outputs=output)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=32)

    return model