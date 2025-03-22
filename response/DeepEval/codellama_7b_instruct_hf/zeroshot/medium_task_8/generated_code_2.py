import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_shape)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)
    branch_path = Flatten()(branch_path)

    # Fuse the main and branch paths
    merged_path = keras.layers.Add()([main_path, branch_path])

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(merged_path)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model