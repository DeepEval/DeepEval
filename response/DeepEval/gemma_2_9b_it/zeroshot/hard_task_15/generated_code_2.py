import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    main_output = layers.Reshape((32, 32, 32))(x)

    # Branch path
    branch_output = layers.Lambda(lambda x: x)(input_layer)

    # Combine outputs
    combined_output = layers.Add()([main_output, branch_output])
    
    # Final classification layers
    x = layers.Flatten()(combined_output)
    x = layers.Dense(units=128, activation='relu')(x)
    output_layer = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model