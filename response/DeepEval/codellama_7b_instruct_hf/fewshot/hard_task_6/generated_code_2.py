import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, Activation, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(main_path)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Define the branch path
    branch_path = Input(shape=input_shape)
    branch_path = Conv2D(32, (1, 1), activation='relu')(branch_path)
    branch_path = Activation('relu')(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Activation('relu')(branch_path)
    branch_path = Conv2D(128, (3, 3), activation='relu')(branch_path)
    branch_path = Activation('relu')(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Merge the main and branch paths
    output = Add()([main_path, branch_path])

    # Define the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=output)

    return model