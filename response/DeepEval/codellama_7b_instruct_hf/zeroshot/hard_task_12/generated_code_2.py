import keras
from keras import layers
from keras.models import Model

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 64))

    # Main path
    x = layers.Conv2D(filters=32, kernel_size=1, strides=1, activation='relu')(input_layer)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(x)
    main_path = layers.Concatenate(axis=-1)(x, x)

    # Branch path
    branch_input = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(input_layer)
    branch_path = layers.Concatenate(axis=-1)(branch_input, branch_input)

    # Combine main and branch paths
    combined_path = layers.Add()([main_path, branch_path])

    # Flatten and classify
    x = layers.Flatten()(combined_path)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model