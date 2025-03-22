from keras.models import Model
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Main path
    main_path = Sequential()
    main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    main_path.add(SeparableConv2D(32, (3, 3), activation='relu'))
    main_path.add(MaxPooling2D((2, 2)))

    # Branch path
    branch_path = Sequential()
    branch_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    branch_path.add(SeparableConv2D(32, (3, 3), activation='relu'))
    branch_path.add(MaxPooling2D((2, 2)))

    # Merge main and branch paths
    merged_path = Add()([main_path.output, branch_path.output])

    # Flatten and output
    flattened_output = Flatten()(merged_path)
    output = Dense(10, activation='softmax')(flattened_output)

    model = Model(inputs=main_path.input, outputs=output)

    return model