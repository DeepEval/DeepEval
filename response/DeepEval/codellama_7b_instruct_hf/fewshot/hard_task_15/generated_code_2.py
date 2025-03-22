import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten, Add

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(main_path)
    x = Dense(64, activation='relu')(x)
    x = Reshape(target_shape=input_shape)(x)
    x = Dense(64, activation='relu')(x)

    # Define the branch path
    branch_path = Input(shape=input_shape)

    # Define the output layer
    output_layer = Add()([main_path, branch_path])
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Define the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=output_layer)

    return model