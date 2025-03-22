from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Dropout

def dl_model():
    # Define the main path of the model
    main_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(main_input)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Define the branch path of the model
    branch_input = Input(shape=(28, 28, 1))
    branch_x = Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_x = Dropout(0.2)(branch_x)
    branch_x = Conv2D(64, (3, 3), activation='relu')(branch_x)
    branch_x = Dropout(0.2)(branch_x)
    branch_x = Conv2D(128, (3, 3), activation='relu')(branch_x)
    branch_x = Flatten()(branch_x)
    branch_x = Dense(128, activation='relu')(branch_x)
    branch_x = Dropout(0.5)(branch_x)
    branch_x = Dense(10, activation='softmax')(branch_x)

    # Add the main and branch paths and output the final probabilities
    output = Add()([main_x, branch_x])
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs=[main_input, branch_input], outputs=output)
    return model