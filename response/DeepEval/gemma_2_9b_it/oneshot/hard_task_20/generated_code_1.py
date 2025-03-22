import tensorflow as tf
from tensorflow.keras import layers

def dl_model():     

    input_layer = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
    
    conv1x1_branch = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    conv3x3_branch = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])
    conv5x5_branch = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x[2])
    
    main_path_output = layers.Concatenate()([conv1x1_branch, conv3x3_branch, conv5x5_branch])

    # Branch Path
    branch_path = layers.Conv2D(filters=192, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fusion
    fused_features = layers.Add()([main_path_output, branch_path])

    # Classification
    x = layers.Flatten()(fused_features)
    x = layers.Dense(units=128, activation='relu')(x)
    output_layer = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model