import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path, main_path])

    # Define the branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)

    # Define the fused features
    fused_features = Concatenate()([main_path, branch_path])

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(fused_features)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


model.fit(x_train, y_train, epochs=10, batch_size=32)