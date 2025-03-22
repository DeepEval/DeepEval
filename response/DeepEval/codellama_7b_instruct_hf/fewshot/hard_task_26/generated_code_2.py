import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, BatchNormalization, Dense, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = input_layer
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dropout(0.2)(main_path)

    # Branch path
    branch_path = input_layer
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dropout(0.2)(branch_path)

    # Merge main and branch path
    output_layer = Concatenate()([main_path, branch_path])
    output_layer = Dense(10, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model