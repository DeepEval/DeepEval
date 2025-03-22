import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        x = GlobalAveragePooling2D()(input_tensor)
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)
        weights = x
        reshaped_weights = keras.backend.reshape(weights, (-1, 32, 1, 1))
        weighted_features = Multiply()([input_tensor, reshaped_weights])
        return weighted_features

    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

    block2_output = block2(block1_output)

    # Branch from Block 1
    branch_output = block1(block2_output)

    # Fusion through addition
    fused_output = Add()([block1_output, branch_output])

    # Final classification
    x = Flatten()(fused_output)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model