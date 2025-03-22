import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu')(conv2)
    bn1 = BatchNormalization()(conv3)
    block1_output = Flatten()(bn1)

    # Block 2
    conv4 = Conv2D(64, (3, 3), activation='relu')(block1_output)
    conv5 = Conv2D(64, (3, 3), activation='relu')(conv4)
    conv6 = Conv2D(64, (3, 3), activation='relu')(conv5)
    bn2 = BatchNormalization()(conv6)
    block2_output = Flatten()(bn2)

    # Block 3
    conv7 = Conv2D(128, (3, 3), activation='relu')(block2_output)
    conv8 = Conv2D(128, (3, 3), activation='relu')(conv7)
    conv9 = Conv2D(128, (3, 3), activation='relu')(conv8)
    bn3 = BatchNormalization()(conv9)
    block3_output = Flatten()(bn3)

    # Final output
    flatten = Flatten()(block3_output)
    output_layer = Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model