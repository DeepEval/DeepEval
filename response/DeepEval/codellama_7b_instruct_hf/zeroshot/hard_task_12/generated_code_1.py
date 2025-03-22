from keras.models import Model
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_data)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Add()([main_path, main_path])
    main_path = Flatten()(main_path)

    # Branch path
    branch_path = Conv2D(64, (3, 3), activation='relu')(input_data)
    branch_path = Flatten()(branch_path)

    # Combine main and branch paths
    output = Add()([main_path, branch_path])

    # Add fully connected layers for classification
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=input_data, outputs=output)

    return model