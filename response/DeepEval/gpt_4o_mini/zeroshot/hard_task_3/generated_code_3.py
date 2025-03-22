import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_tensor = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Process each split through 1x1 and 3x3 convolutions
    processed_splits = []
    for split in splits:
        x = Conv2D(32, (1, 1), activation='relu')(split)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        processed_splits.append(x)

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(processed_splits)

    # Parallel branch processing
    branch_pathway = Conv2D(64, (1, 1), activation='relu')(input_tensor)

    # Combine main pathway and branch pathway
    combined = Add()([main_pathway, branch_pathway])

    # Fully connected layer for classification
    x = Flatten()(combined)
    x = Dense(128, activation='relu')(x)
    output_tensor = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()  # To view the model architecture