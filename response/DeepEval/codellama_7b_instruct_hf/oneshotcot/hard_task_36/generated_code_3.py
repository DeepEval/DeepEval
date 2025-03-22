import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import Dropout
from keras.models import Model

def dl_model():
    # Main pathway
    main_input = Input(shape=(28, 28, 1))
    main_conv1 = Conv2D(32, (3, 3), activation='relu')(main_input)
    main_conv2 = Conv2D(32, (1, 1), activation='relu')(main_conv1)
    main_maxpool = MaxPooling2D(pool_size=(2, 2))(main_conv2)
    main_dropout = Dropout(0.5)(main_maxpool)

    # Branch pathway
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_conv2 = Conv2D(32, (1, 1), activation='relu')(branch_conv)
    branch_output = Conv2D(32, (5, 5), activation='relu')(branch_conv2)

    # Fuse outputs
    fuse_layer = Concatenate()([main_dropout, branch_output])

    # Global average pooling
    gavgpool = GlobalAveragePooling2D()(fuse_layer)

    # Flatten
    flatten = Flatten()(gavgpool)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create and compile model
    model = Model(inputs=[main_input, branch_input], outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model