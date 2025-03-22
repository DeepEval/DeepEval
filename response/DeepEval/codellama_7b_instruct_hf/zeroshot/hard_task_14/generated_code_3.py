import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, Add
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(main_path)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Define the branch path
    branch_path = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(branch_path)
    x = Conv2D(32, (3, 3), activation='relu')(x)

    # Add the main and branch paths
    x = Add()([main_path, branch_path])

    # Define the final layers
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[main_path, branch_path], outputs=x)

    return model