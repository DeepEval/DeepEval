import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ReLU

def dl_model():
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        batch_norm = BatchNormalization()(conv3)
        pool = MaxPooling2D(pool_size=(2, 2))(batch_norm)
        return pool

    input_layer = Input(shape=(32, 32, 3))
    output_layer = block(input_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()