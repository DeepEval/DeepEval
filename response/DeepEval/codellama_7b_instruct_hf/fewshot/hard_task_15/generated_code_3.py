from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten, Add

 和 return model
def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(64, activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine main and branch paths
    combined_path = Add()([main_path, branch_path])

    # Fully connected layers
    combined_path = Flatten()(combined_path)
    combined_path = Dense(128, activation='relu')(combined_path)
    combined_path = Dense(64, activation='relu')(combined_path)
    combined_path = Dense(10, activation='softmax')(combined_path)

    model = Model(inputs=input_layer, outputs=combined_path)

    return model