from keras.applications import VGG16
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Add, BatchNormalization
from keras.models import Model

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    main_x = GlobalAveragePooling2D()(main_input)
    main_x = Flatten()(main_x)
    main_x = Dense(128, activation='relu')(main_x)
    main_x = Dense(64, activation='relu')(main_x)
    main_x = Dense(10, activation='softmax')(main_x)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_x = Flatten()(branch_x)
    branch_x = Dense(64, activation='relu')(branch_x)
    branch_x = Dense(10, activation='softmax')(branch_x)

    # Combine main and branch paths
    merged_output = Add()([main_x, branch_x])

    # Final fully connected layers
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(10, activation='softmax')(merged_output)

    model = Model(inputs=[main_input, branch_input], outputs=merged_output)

    return model