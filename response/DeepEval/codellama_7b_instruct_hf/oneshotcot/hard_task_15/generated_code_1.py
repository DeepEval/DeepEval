import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    main_path = x

    # Branch path
    branch_path = input_layer

    # Combine main and branch paths
    x = Concatenate()([main_path, branch_path])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model