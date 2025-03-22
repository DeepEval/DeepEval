import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16


def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(16, (3, 3), activation='relu')(input_layer)


    def block(input_tensor):
        path1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(64, (3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(64, (5, 5), activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor


    def block_2(input_tensor):
        path1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(64, (3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(64, (5, 5), activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor


    def feature_fusion(input_tensor):
        path1 = Conv2D(128, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(128, (3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(128, (5, 5), activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor


    output_layer = Dense(10, activation='softmax')(output_tensor)


    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model