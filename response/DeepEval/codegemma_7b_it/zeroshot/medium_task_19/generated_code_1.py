from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # First branch: 1x1 convolution for dimensionality reduction
    branch_1 = Conv2D(64, (1, 1), activation='relu')(input_img)

    # Second branch: 1x1 followed by 3x3 convolution
    branch_2 = Conv2D(64, (1, 1), activation='relu')(input_img)
    branch_2 = Conv2D(64, (3, 3), activation='relu')(branch_2)

    # Third branch: 1x1 followed by 5x5 convolution
    branch_3 = Conv2D(64, (1, 1), activation='relu')(input_img)
    branch_3 = Conv2D(64, (5, 5), activation='relu')(branch_3)

    # Fourth branch: 3x3 max pooling followed by 1x1 convolution
    branch_4 = MaxPooling2D((3, 3), strides=(2, 2))(input_img)
    branch_4 = Conv2D(64, (1, 1), activation='relu')(branch_4)

    # Concatenate the outputs of all branches
    concat_layers = concatenate([branch_1, branch_2, branch_3, branch_4])

    # Flatten the features
    flatten_features = Flatten()(concat_layers)

    # Fully connected layers for classification
    dense_layer_1 = Dense(256, activation='relu')(flatten_features)
    output_layer = Dense(10, activation='softmax')(dense_layer_1)

    # Construct the model
    model = Model(inputs=input_img, outputs=output_layer)

    return model