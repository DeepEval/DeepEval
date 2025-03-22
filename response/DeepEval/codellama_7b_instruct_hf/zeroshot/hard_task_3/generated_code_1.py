import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Branch pathway
    branch_pathway = layers.Conv2D(64, (1, 1), activation='relu')(x[0])
    branch_pathway = layers.Conv2D(64, (3, 3), activation='relu')(branch_pathway)
    branch_pathway = layers.Dropout(0.2)(branch_pathway)

    # Main pathway
    main_pathway = layers.Conv2D(64, (1, 1), activation='relu')(x[1])
    main_pathway = layers.Conv2D(64, (3, 3), activation='relu')(main_pathway)
    main_pathway = layers.Dropout(0.2)(main_pathway)

    # Combine branch and main pathways
    combined_pathway = layers.Add()([branch_pathway, main_pathway])

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(combined_pathway)

    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model