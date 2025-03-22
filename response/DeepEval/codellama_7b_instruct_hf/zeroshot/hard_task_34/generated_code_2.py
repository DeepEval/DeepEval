from keras.layers import Input, Dense, Flatten, Conv2D, SeparableConv2D, ReLU, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Add()([main_path, input_layer])
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Add()([main_path, input_layer])
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Add()([main_path, input_layer])
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = Add()([branch_path, input_layer])
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(branch_path)
    branch_path = Add()([branch_path, input_layer])
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)

    # Combine features
    output_layer = Add()([main_path, branch_path])

    # Define model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model