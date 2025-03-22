from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block_1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    block_1 = BatchNormalization()(block_1)

    # Define the second block
    block_2 = Conv2D(64, (3, 3), activation='relu')(block_1)
    block_2 = BatchNormalization()(block_2)
    block_2 = Add()([block_1, block_2])

    # Define the feature fusion
    feature_fusion = Add()([block_1, block_2])

    # Define the main structure
    main_structure = Conv2D(16, (3, 3), activation='relu')(feature_fusion)
    main_structure = BatchNormalization()(main_structure)

    # Define the branch
    branch = Conv2D(16, (3, 3), activation='relu')(main_structure)
    branch = BatchNormalization()(branch)

    # Define the output
    output = Add()([main_structure, branch])
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model