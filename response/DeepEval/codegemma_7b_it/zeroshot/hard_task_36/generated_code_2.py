from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, concatenate, add, Flatten, Dense, GlobalAveragePooling2D

def residual_block(input_tensor, filters):
    # Main pathway
    conv = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)

    conv = Conv2D(filters, (1, 1), padding='same')(act)
    bn = BatchNormalization()(conv)

    # Branch pathway
    conv_branch = Conv2D(filters, (1, 1), padding='same')(input_tensor)
    bn_branch = BatchNormalization()(conv_branch)

    # Fuse outputs
    add_path = add([act, bn_branch])
    act_fused = Activation('relu')(add_path)

    return act_fused

def dl_model():

    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Main pathway
    conv = Conv2D(32, (3, 3), padding='same')(inputs)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)

    conv = Conv2D(32, (3, 3), padding='same')(act)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    max_pool = MaxPooling2D(pool_size=(2, 2))(act)
    drop = Dropout(0.5)(max_pool)

    # Residual block
    residual = residual_block(drop, 32)

    # Branch pathway
    conv_branch = Conv2D(32, (1, 1), padding='same')(max_pool)
    bn_branch = BatchNormalization()(conv_branch)
    max_pool_branch = MaxPooling2D(pool_size=(2, 2))(bn_branch)
    drop_branch = Dropout(0.5)(max_pool_branch)

    # Fuse outputs
    add_path = add([residual, drop_branch])
    act_fused = Activation('relu')(add_path)

    # Output layer
    gap = GlobalAveragePooling2D()(act_fused)
    flat = Flatten()(gap)
    outputs = Dense(10, activation='softmax')(flat)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model