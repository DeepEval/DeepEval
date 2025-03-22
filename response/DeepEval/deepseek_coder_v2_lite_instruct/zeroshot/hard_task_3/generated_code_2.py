import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_tensor = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

    # Main pathway
    main_pathway = []
    for group in split:
        # 1x1 convolution
        x = Conv2D(32, (1, 1), activation='relu')(group)
        # 3x3 convolution
        x = Conv2D(32, (3, 3), activation='relu')(x)
        main_pathway.append(x)

    # Dropout for feature selection
    main_pathway = [Dropout(0.25)(x) for x in main_pathway]

    # Concatenate the outputs from the three groups
    main_output = tf.concat(main_pathway, axis=-1)

    # Branch pathway
    branch_pathway = Conv2D(32, (1, 1), activation='relu')(input_tensor)

    # Add the main pathway and branch pathway
    combined = Add()([main_output, branch_pathway])

    # Flatten the output
    flatten = Flatten()(combined)

    # Fully connected layer
    output = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()