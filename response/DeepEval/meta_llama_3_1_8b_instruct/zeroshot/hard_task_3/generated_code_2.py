# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model using Functional API
    inputs = keras.Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    channel_axis, spatial_axis = (-1, 1), (1, 2)
    split_input = layers.Lambda(
        lambda x: tf.split(x, num_or_size_splits=3, axis=channel_axis))(inputs)

    # Define the main pathway
    main_pathway = []
    for i in range(3):
        # Extract deep features
        x = layers.Conv2D(32, (1, 1), activation='relu')(split_input[i])
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        
        # Feature selection to mitigate overfitting
        x = layers.Dropout(0.2)(x)
        
        # Append to the main pathway
        main_pathway.append(x)

    # Concatenate the outputs from the three groups
    main_output = layers.Concatenate()(main_pathway)

    # Define the branch pathway
    branch_output = layers.Conv2D(64, (1, 1), activation='relu')(inputs)

    # Combine the outputs from both pathways
    combined_output = layers.Add()([main_output, branch_output])

    # Flatten the output
    flattened_output = layers.Flatten()(combined_output)

    # Final classification layer
    outputs = layers.Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Compile the model
def compile_model(model):
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=['accuracy']
    )
    return model

# Create the model and compile it
model = dl_model()
model = compile_model(model)

# Print the model summary
model.summary()