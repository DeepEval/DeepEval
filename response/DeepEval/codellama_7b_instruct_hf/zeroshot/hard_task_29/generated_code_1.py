import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Add

def dl_model():
    # Main block
    main_input = Input(shape=(28, 28, 1))
    main_conv1 = Conv2D(32, (3, 3), activation='relu')(main_input)
    main_conv2 = Conv2D(64, (3, 3), activation='relu')(main_conv1)
    main_output = Add()([main_conv1, main_conv2])

    # Branch block
    branch_input = Input(shape=(28, 28, 1))
    branch_pool1 = MaxPooling2D(pool_size=(1, 1))(branch_input)
    branch_pool2 = MaxPooling2D(pool_size=(2, 2))(branch_pool1)
    branch_pool3 = MaxPooling2D(pool_size=(4, 4))(branch_pool2)
    branch_output = Add()([branch_pool1, branch_pool2, branch_pool3])

    # Combine main and branch outputs
    output = Add()([main_output, branch_output])

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create and return model
    model = Model(inputs=[main_input, branch_input], outputs=output)
    return model