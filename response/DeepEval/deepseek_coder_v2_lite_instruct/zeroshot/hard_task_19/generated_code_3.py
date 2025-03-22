import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(main_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the branch path
    branch_input = Input(shape=input_shape)
    y = Conv2D(32, (1, 1), activation='relu')(branch_input)
    y = GlobalAveragePooling2D()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(input_shape[-1], activation='sigmoid')(y)
    y = Multiply()([branch_input, tf.reshape(y, (-1, 1, 1, input_shape[-1]))])

    # Add the outputs from both paths
    combined = Add()([x, y])

    # Additional fully connected layers for classification
    z = GlobalAveragePooling2D()(combined)
    z = Dense(128, activation='relu')(z)
    output = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=output)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()