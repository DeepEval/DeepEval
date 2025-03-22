import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.models import Model


def dl_model():
    # Define the input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the main path
    main_path = Input(shape=input_shape)
    main_path_splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_path)
    main_path_1 = main_path_splits[0]
    main_path_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_path_splits[1])
    main_path_3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_path_2)
    main_path_output = Add()([main_path_1, main_path_3])

    # Define the branch path
    branch_path = Input(shape=input_shape)
    branch_path_1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(branch_path)
    branch_path_output = branch_path_1

    # Define the fusion layer
    fusion_layer = Add()([main_path_output, branch_path_output])

    # Define the final classification layer
    final_layer = Flatten()(fusion_layer)
    final_layer = Dense(num_classes, activation='softmax')(final_layer)

    # Create the model
    model = Model(inputs=[main_path, branch_path], outputs=final_layer)

    return model