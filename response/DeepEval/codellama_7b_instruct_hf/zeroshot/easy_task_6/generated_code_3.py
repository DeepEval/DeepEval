from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Main path
    main_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(main_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    main_output = x

    # Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_x = Conv2D(16, (3, 3), activation='relu')(branch_input)
    branch_x = Conv2D(32, (3, 3), activation='relu')(branch_x)
    branch_x = MaxPooling2D((2, 2))(branch_x)
    branch_x = Flatten()(branch_x)
    branch_x = Dense(64, activation='relu')(branch_x)
    branch_x = Dense(10, activation='softmax')(branch_x)
    branch_output = branch_x

    # Combine main and branch paths
    x = main_output + branch_output

    # Create model
    model = Model(inputs=[main_input, branch_input], outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model