from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, BatchNormalization
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = input_layer
    for i in range(2):
        main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
        main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
        main_path = MaxPooling2D((2, 2))(main_path)

    # Branch path
    branch_path = Conv2D(16, (1, 1), activation='relu')(input_layer)

    # Combine main and branch paths
    combined_path = Add()([main_path, branch_path])

    # Flatten and fully connected layer
    flattened = Flatten()(combined_path)
    output = Dense(10, activation='softmax')(flattened)

    # Define model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model