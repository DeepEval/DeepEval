import keras
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    def block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        relu = ReLU()(bn)
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        return conv, relu

    def pathway1(input_tensor):
        conv1, relu1 = block(input_tensor)
        conv2, relu2 = block(relu1)
        conv3, relu3 = block(relu2)
        concat = Concatenate()([conv1, conv2, conv3])
        return Flatten()(concat)

    # Pathway 2
    def block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        relu = ReLU()(bn)
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        return conv, relu

    def pathway2(input_tensor):
        conv1, relu1 = block(input_tensor)
        conv2, relu2 = block(relu1)
        conv3, relu3 = block(relu2)
        concat = Concatenate()([conv1, conv2, conv3])
        return Flatten()(concat)

    pathway1_output = pathway1(input_tensor=input_layer)
    pathway2_output = pathway2(input_tensor=input_layer)

    # Merge pathways
    merge = Concatenate()([pathway1_output, pathway2_output])

    # Classification
    dense1 = Dense(units=128, activation='relu')(merge)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])