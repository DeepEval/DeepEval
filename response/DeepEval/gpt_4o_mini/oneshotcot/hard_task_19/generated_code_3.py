import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Add, Multiply, Reshape, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Construct the main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Step 3: Construct the branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=32, activation='sigmoid')(branch_path)

    # Step 4: Reshape and multiply the weights with the input
    branch_path = Reshape((1, 1, 32))(branch_path)
    weighted_input = Multiply()([input_layer, branch_path])

    # Step 5: Combine the outputs from main and branch paths
    combined_output = Add()([main_path, weighted_input])

    # Step 6: Flatten the combined output
    flatten_output = Flatten()(combined_output)

    # Step 7: Add fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Step 8: Create the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model