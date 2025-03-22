import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def group_convolutions(input_tensor):
        x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
        x3 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_tensor)
        return Concatenate()([x1, x2, x3])
    
    group_outputs = Lambda(group_convolutions)(input_layer)

    # Second block
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(group_outputs)
    branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(group_outputs)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(group_outputs)
    branch3 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch3)

    # Concatenate outputs from all branches
    concat_outputs = Concatenate()([branch1, branch2, branch3])

    # Global average pooling and fully connected layer
    avg_pool = GlobalAveragePooling2D()(concat_outputs)
    output_layer = Dense(10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model