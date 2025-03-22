import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(path1)
    path2 = Conv2D(64, (2, 2), activation='relu')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path2)
    path3 = Conv2D(64, (4, 4), activation='relu')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(path3)

    # Concatenate outputs of the three paths
    concat_layer = Concatenate(axis=-1)([Flatten()(path1), Flatten()(path2), Flatten()(path3)])
    concat_layer = Dropout(0.5)(concat_layer)

    # Transform the output to 4-dimensional tensor format
    reshape_layer = Reshape((1, 1, concat_layer.shape[1]))(concat_layer)

    # Block 2
    path1_block2 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    path2_block2 = Conv2D(64, (1, 7), activation='relu')(Conv2D(64, (1, 1), activation='relu')(input_layer))
    path3_block2 = Conv2D(64, (7, 1), activation='relu')(Conv2D(64, (1, 1), activation='relu')(input_layer))
    path4_block2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path4_block2 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7))(path4_block2)

    # Concatenate outputs of the four paths along the channel dimension
    concat_block2 = Concatenate(axis=-1)([path1_block2, path2_block2, path3_block2, path4_block2])

    # Flatten the concatenated result
    flatten_layer = Flatten()(concat_block2)
    flatten_layer = Dropout(0.5)(flatten_layer)

    # Output layer
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model