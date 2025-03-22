import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Define the main pathway
    def pathway(input_tensor):
        # First apply 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # Then apply 3x3 convolution
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1)
        return path2

    # Apply the pathway to each group
    paths = [pathway(split_layer[i]) for i in range(3)]

    # Dropout for regularization
    paths = [Dropout(0.25)(path) for path in paths]

    # Concatenate the outputs from the three groups
    main_path = Add()(paths)

    # Parallel branch pathway
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the main pathway and the branch pathway
    combined = Add()([main_path, branch_path])

    # Flatten the result
    flatten_layer = Flatten()(combined)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])