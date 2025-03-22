from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    residual_block_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    residual_block_1 = BatchNormalization()(residual_block_1)
    residual_block_1 = ReLU()(residual_block_1)
    residual_block_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(residual_block_1)
    residual_block_1 = BatchNormalization()(residual_block_1)
    residual_block_1 = ReLU()(residual_block_1)
    x = Add()([x, residual_block_1])

    residual_block_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    residual_block_2 = BatchNormalization()(residual_block_2)
    residual_block_2 = ReLU()(residual_block_2)
    residual_block_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(residual_block_2)
    residual_block_2 = BatchNormalization()(residual_block_2)
    residual_block_2 = ReLU()(residual_block_2)
    x = Add()([x, residual_block_2])

    global_branch = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    x = GlobalAveragePooling2D()(global_branch)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model