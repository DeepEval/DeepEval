from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the main path
    main_input = Input(shape=input_shape)
    conv1 = Conv2D(32, (1, 1), activation='relu')(main_input)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    main_output = Concatenate()([conv1, conv2])

    # Define the branch path
    branch_input = Input(shape=input_shape)
    conv3 = Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_output = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=Add()([main_output, branch_output]))
    model.summary()

    return model