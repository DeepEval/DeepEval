from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Reshape, Multiply, Add
from keras.models import Model

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    main_conv1 = Conv2D(32, (3, 3), activation='relu')(main_input)
    main_conv2 = Conv2D(64, (3, 3), activation='relu')(main_conv1)
    main_conv3 = Conv2D(64, (3, 3), activation='relu')(main_conv2)
    main_maxpool = MaxPooling2D((2, 2))(main_conv3)
    main_fc1 = Dense(128, activation='relu')(main_maxpool)
    main_fc2 = Dense(10, activation='softmax')(main_fc1)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_gap = GlobalAveragePooling2D()(branch_input)
    branch_fc1 = Dense(64, activation='relu')(branch_gap)
    branch_fc2 = Dense(10, activation='softmax')(branch_fc1)

    # Merge the paths
    merged_input = Concatenate()([main_fc2, branch_fc2])
    merged_fc1 = Dense(128, activation='relu')(merged_input)
    merged_fc2 = Dense(10, activation='softmax')(merged_fc1)

    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=merged_fc2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model