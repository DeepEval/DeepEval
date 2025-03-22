from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the initial 1x1 convolutional layer
    initial_conv = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the first branch: local feature extraction
    local_feature_extractor = Conv2D(32, (3, 3), activation='relu')(initial_conv)
    local_feature_extractor = Conv2D(32, (3, 3), activation='relu')(local_feature_extractor)

    # Define the second branch: downsampling
    downsampled_feature_extractor = AveragePooling2D((2, 2))(initial_conv)
    downsampled_feature_extractor = Conv2D(32, (3, 3), activation='relu')(downsampled_feature_extractor)
    downsampled_feature_extractor = AveragePooling2D((2, 2))(downsampled_feature_extractor)

    # Define the third branch: upsampling
    upsampled_feature_extractor = Conv2DTranspose(32, (3, 3), activation='relu')(initial_conv)
    upsampled_feature_extractor = Conv2DTranspose(32, (3, 3), activation='relu')(upsampled_feature_extractor)

    # Concatenate the outputs of the different branches
    concatenated_feature_extractor = Concatenate()([local_feature_extractor, downsampled_feature_extractor, upsampled_feature_extractor])

    # Refine the concatenated feature extractor
    refined_feature_extractor = Conv2D(32, (1, 1), activation='relu')(concatenated_feature_extractor)

    # Flatten the refined feature extractor and pass it through a fully connected layer
    flattened_refined_feature_extractor = Flatten()(refined_feature_extractor)
    output_layer = Dense(10, activation='softmax')(flattened_refined_feature_extractor)

    # Define the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model