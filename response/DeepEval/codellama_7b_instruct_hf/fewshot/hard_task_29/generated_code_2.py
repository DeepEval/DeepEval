from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    main_path = Add()([conv1, conv2])

    # Branch path
    branch_path = input_layer

    # Combine main and branch paths
    add_layer = Add()([main_path, branch_path])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(64, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model