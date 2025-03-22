import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
    pool = MaxPooling2D((2, 2))(conv3)

    # Branch path
    branch_conv1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch_conv2 = Conv2D(128, (3, 3), activation='relu')(branch_conv1)
    branch_pool = GlobalAveragePooling2D()(branch_conv2)
    branch_fc1 = Dense(64, activation='relu')(branch_pool)
    branch_fc2 = Dense(10, activation='softmax')(branch_fc1)

    # Merge the outputs
    merged_output = Add()([pool, branch_fc2])

    # Flatten and dense layers
    flatten = Flatten()(merged_output)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model