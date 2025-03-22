import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)

    merged = Add()([branch1, branch2])

    pooled = GlobalAveragePooling2D()(merged)

    attention1 = Dense(units=1, activation='softmax')(pooled)
    attention2 = Dense(units=1, activation='softmax')(pooled)

    weighted_branch1 = merged * attention1
    weighted_branch2 = merged * attention2

    final_output = Add()([weighted_branch1, weighted_branch2])

    output_layer = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model