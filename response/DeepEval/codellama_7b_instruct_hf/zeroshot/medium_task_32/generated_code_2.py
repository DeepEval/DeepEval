import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    groups = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 3})(input_layer)

    # Extract features from each group
    group_1 = Conv2D(32, (1, 1), activation='relu')(groups[0])
    group_2 = Conv2D(64, (3, 3), activation='relu')(groups[1])
    group_3 = Conv2D(128, (5, 5), activation='relu')(groups[2])

    # Concatenate the output of each group
    fused_features = tf.concat([group_1, group_2, group_3], axis=1)

    # Flatten the fused features
    flattened_features = Flatten()(fused_features)

    # Pass the flattened features through a fully connected layer
    output = Dense(10, activation='softmax')(flattened_features)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model