from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, Flatten, Add
from keras.models import Model

def dl_model():
    # Define the main and branch paths
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    branch_path = Conv2D(16, (3, 3), activation='relu')(input_layer)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = GlobalAveragePooling2D()(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Add the main and branch paths
    merged_path = Add()([main_path, branch_path])

    # Flatten the merged path and add two fully connected layers
    merged_path = Flatten()(merged_path)
    merged_path = Dense(128, activation='relu')(merged_path)
    merged_path = Dense(10, activation='softmax')(merged_path)

    # Create the model
    model = Model(inputs=input_layer, outputs=merged_path)

    return model