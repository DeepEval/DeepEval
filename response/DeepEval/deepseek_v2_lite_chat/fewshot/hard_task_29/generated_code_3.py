import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Dense

def dl_model():
    # Block 1: Main path and branch path
    main_input = Input(shape=(28, 28, 1))
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_input)
    conv1_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1_main)
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_input)

    # Branch path connects directly to the input
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_input)

    # Add main path and branch path to get the output of Block 1
    add_layer = Add()([conv1_main, branch_path])

    # Block 2: Three max pooling layers with varying scales
    pool1_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(add_layer)
    pool2_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(add_layer)
    pool3_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(add_layer)

    # Flatten and pass through fully connected layers for final classification
    flat1 = Flatten()(pool1_1x1)
    flat2 = Flatten()(pool2_2x2)
    flat3 = Flatten()(pool3_4x4)

    concat = Concatenate()([flat1, flat2, flat3])
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=main_input, outputs=output_layer)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])