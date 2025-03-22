import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)  # Global Average Pooling
    branch_path = Dense(units=128, activation='relu')(branch_path)  # Fully connected layer
    branch_path = Dense(units=96, activation='relu')(branch_path)  # Fully connected layer to generate weights
    branch_weights = Dense(units=32 * 32 * 32, activation='sigmoid')(branch_path)  # Output weights

    # Reshape weights for multiplication
    branch_weights = keras.layers.Reshape((32, 32, 32))(branch_weights)

    # Multiply branch weights with main path output
    multiplied_output = Multiply()([main_path, branch_weights])

    # Combine both paths
    combined_output = Add()([main_path, multiplied_output])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model