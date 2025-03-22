import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Concatenate, BatchNormalization
from keras.applications import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path of the model
    main_path = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(main_path)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the branch path of the model
    branch_path = Input(shape=input_shape)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Combine the main and branch paths
    x = Concatenate()([main_path, branch_path])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create and return the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=x)
    return model