import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(filters=3, kernel_size=3, padding='same')(inputs)

    # Block 1: Channel Attention
    path1 = layers.GlobalAveragePooling2D()(x)
    path1 = layers.Dense(units=1024, activation='relu')(path1)
    path1 = layers.Dense(units=3, activation='sigmoid')(path1)
    path2 = layers.GlobalMaxPooling2D()(x)
    path2 = layers.Dense(units=1024, activation='relu')(path2)
    path2 = layers.Dense(units=3, activation='sigmoid')(path2)
    attention_path = layers.Add()([path1, path2])
    attention_path = layers.Multiply()([x, attention_path])

    # Block 2: Spatial Attention
    avg_pool = layers.AveragePooling2D()(x)
    max_pool = layers.MaxPooling2D()(x)
    concat_pool = layers.concatenate([avg_pool, max_pool])
    concat_pool = layers.Conv2D(filters=3, kernel_size=1, activation='sigmoid')(concat_pool)
    concat_pool = layers.Multiply()([x, concat_pool])

    # Main path
    main_path = layers.Add()([attention_path, concat_pool])
    main_path = layers.Activation('relu')(main_path)
    main_path = layers.Conv2D(filters=3, kernel_size=3, padding='same')(main_path)

    # Output layer
    outputs = layers.Dense(units=10, activation='softmax')(main_path)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model