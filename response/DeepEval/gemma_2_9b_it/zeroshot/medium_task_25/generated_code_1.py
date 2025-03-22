import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Path 1
    path1_output = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(input_tensor)

    # Path 2
    path2_output = layers.AveragePooling2D(pool_size=(2, 2))(input_tensor)
    path2_output = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(path2_output)

    # Path 3
    path3_output1 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(input_tensor)
    path3_output2 = layers.Conv2D(filters=16, kernel_size=(1, 3), activation='relu')(path3_output1)
    path3_output3 = layers.Conv2D(filters=16, kernel_size=(3, 1), activation='relu')(path3_output1)
    path3_output = layers.concatenate([path3_output2, path3_output3], axis=-1)

    # Path 4
    path4_output1 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(input_tensor)
    path4_output2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(path4_output1)
    path4_output3 = layers.Conv2D(filters=16, kernel_size=(1, 3), activation='relu')(path4_output2)
    path4_output4 = layers.Conv2D(filters=16, kernel_size=(3, 1), activation='relu')(path4_output2)
    path4_output = layers.concatenate([path4_output3, path4_output4], axis=-1)

    # Feature Fusion
    output = layers.concatenate([path1_output, path2_output, path3_output, path4_output], axis=-1)

    # Classification
    output = layers.Flatten()(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = models.Model(inputs=input_tensor, outputs=output)

    return model