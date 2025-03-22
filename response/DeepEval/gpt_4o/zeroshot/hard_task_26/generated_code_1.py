from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1 - 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    # Branch 2 - max pooling, 3x3 convolutional, upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3 - max pooling, 3x3 convolutional, upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenating the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer after concatenation
    main_path_output = Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Branch path - starts with a 1x1 convolutional layer
    branch_path = Conv2D(32, (1, 1), activation='relu')(x)

    # Adding the outputs from main path and branch path
    added_output = Add()([main_path_output, branch_path])

    # Fully connected layers for classification
    flattened = Flatten()(added_output)
    fc1 = Dense(256, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model