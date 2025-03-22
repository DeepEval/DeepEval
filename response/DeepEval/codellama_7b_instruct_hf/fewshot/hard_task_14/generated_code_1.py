import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Add, Conv2D
from keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(32, activation='relu')(main_path)

    # Define branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)

    # Merge main and branch paths
    merged_path = Add()([main_path, branch_path])

    # Pass through fully connected layers
    merged_path = Flatten()(merged_path)
    merged_path = Dense(128, activation='relu')(merged_path)
    merged_path = Dense(10, activation='softmax')(merged_path)

    # Define model
    model = Model(inputs=input_layer, outputs=merged_path)

    return model