import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Convolutional layer
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)

    # Step 3: Average Pooling layers
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1)(conv1)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv1)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4)(conv1)

    # Flatten the output of average pooling layers
    flat_1 = Flatten()(avg_pool_1)
    flat_2 = Flatten()(avg_pool_2)
    flat_3 = Flatten()(avg_pool_3)

    # Concatenate the flattened outputs
    concat = Concatenate()( [flat_1, flat_2, flat_3] )

    # Step 4: Fully connected layer and reshape
    dense1 = Dense(units=128, activation='relu')(concat)
    reshape = Reshape((4, 4, 1))(dense1)

    # Step 5: Second block
    split1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=4, axis=0))(reshape)
    split1[0] = Conv2D(64, kernel_size=(1, 1), activation='relu')(split1[0])
    split1[1] = Conv2D(64, kernel_size=(3, 3), activation='relu')(split1[1])
    split1[2] = Conv2D(64, kernel_size=(5, 5), activation='relu')(split1[2])
    split1[3] = AveragePooling2D(pool_size=(1, 1), strides=1)(split1[3])

    concat_output = Concatenate()(split1)
    flatten = Flatten()(concat_output)
    dense2 = Dense(units=128, activation='relu')(flatten)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model