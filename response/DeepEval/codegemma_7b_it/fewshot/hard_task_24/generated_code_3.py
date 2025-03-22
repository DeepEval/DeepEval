import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    # Branch 2
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = UpSampling2D(size=(2, 2))(branch_2)

    # Branch 3
    branch_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(initial_conv)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = UpSampling2D(size=(4, 4))(branch_3)

    # Fusion layers
    merged = concatenate([branch_1, branch_2, branch_3])

    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(merged)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model