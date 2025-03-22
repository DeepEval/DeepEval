from keras.layers import Input, Dense, Conv2D, ReLU, MaxPooling2D, Flatten, concatenate, Add
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = VGG16(include_top=False, input_shape=input_shape, pooling='avg')
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path
    branch_path = Conv2D(64, (3, 3), activation='relu')(input_shape)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Flatten()(branch_path)

    # Define the fusion layer
    fusion_layer = Add()([main_path, branch_path])

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(fusion_layer)

    # Define the model
    model = Model(inputs=input_shape, outputs=output_layer)

    return model