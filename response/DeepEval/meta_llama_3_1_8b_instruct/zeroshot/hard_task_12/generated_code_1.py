from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from keras.models import Model
from keras import backend as K

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the main path
    inputs = Input(shape=input_shape)
    main_path = Conv2D(64, (1, 1), activation='relu', name='main_path_conv1')(inputs)
    main_path = Conv2D(128, (1, 1), activation='relu', name='main_path_conv2')(main_path)
    main_path = Conv2D(128, (3, 3), activation='relu', name='main_path_conv3')(main_path)
    main_path = Conv2D(128, (1, 1), activation='relu', name='main_path_conv4')(main_path)

    # Define the branch path
    branch_path = Conv2D(128, (3, 3), activation='relu', name='branch_path_conv1')(inputs)

    # Combine the main and branch paths
    combined = Add()([main_path, branch_path])

    # Flatten the output
    x = Flatten()(combined)

    # Define the fully connected layers
    x = Dense(128, activation='relu', name='fc1')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.summary()