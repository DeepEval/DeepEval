from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from keras.applications.resnet import ResNet


def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the parallel branch of convolutional layers
    parallel_branch = ResNet(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        classes=10
    )

    # Define the sequential blocks
    block1 = Conv2D(32, (3, 3), activation='relu')(parallel_branch.output)
    block1 = BatchNormalization()(block1)
    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block3 = Conv2D(128, (3, 3), activation='relu')(block2)
    block3 = BatchNormalization()(block3)

    # Define the output paths
    path1 = Flatten()(block1)
    path2 = Flatten()(block2)
    path3 = Flatten()(block3)

    # Define the aggregated output
    aggregated_output = Add()([path1, path2, path3])

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')(aggregated_output)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=parallel_branch.input, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model