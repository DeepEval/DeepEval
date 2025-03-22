import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout1 = Dropout(0.2)(conv2)
        return dropout1

    def branch2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        dropout2 = Dropout(0.2)(conv4)
        return dropout2

    def branch3(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        dropout3 = Dropout(0.2)(maxpool)
        return dropout3

    branch1_output = branch1(input_tensor=input_layer)
    branch2_output = branch2(input_tensor=input_layer)
    branch3_output = branch3(input_tensor=input_layer)

    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def main():
    model = dl_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

if __name__ == "__main__":
    main()