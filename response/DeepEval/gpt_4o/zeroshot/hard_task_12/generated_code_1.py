from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 64)
    inputs = Input(shape=input_shape)

    # Main path
    # 1x1 convolution for dimensionality reduction
    main_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Parallel convolutional layers
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(main_path)
    conv3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)

    # Concatenate the outputs of parallel convolutions
    main_path = Concatenate()([conv1x1, conv3x3])

    # Branch path
    branch_path = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # Combine main and branch paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten = Flatten()(combined)

    # Fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for classification

    # Construct the model
    model = Model(inputs=inputs, outputs=fc2)

    return model