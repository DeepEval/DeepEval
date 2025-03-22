from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Main path
    initial_conv = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)
    branch_1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(initial_conv)

    branch_2 = layers.MaxPooling2D(pool_size=2)(initial_conv)
    branch_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch_2)
    branch_2 = layers.UpSampling2D()(branch_2)

    branch_3 = layers.MaxPooling2D(pool_size=2)(initial_conv)
    branch_3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch_3)
    branch_3 = layers.UpSampling2D()(branch_3)

    merged = layers.concatenate([branch_1, branch_2, branch_3])
    main_path = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(merged)

    # Branch path
    branch_path = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)

    # Combine paths
    combined = layers.add([main_path, branch_path])

    # Classification layers
    flatten = layers.Flatten()(combined)
    dense_1 = layers.Dense(units=64, activation='relu')(flatten)
    outputs = layers.Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=inputs, outputs=outputs)

    return model