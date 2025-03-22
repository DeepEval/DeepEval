import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    input_tensor = layers.Input(shape=(32, 32, 3))
    
    # Split input channels
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)

    # Main Pathway
    output1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(split_tensor[0])
    output1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(output1)
    output1 = layers.Dropout(0.25)(output1)

    output2 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(split_tensor[1])
    output2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(output2)
    output2 = layers.Dropout(0.25)(output2)

    output3 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(split_tensor[2])
    output3 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(output3)
    output3 = layers.Dropout(0.25)(output3)

    # Concatenate outputs from main pathway
    concatenated = layers.Concatenate(axis=3)([output1, output2, output3])

    # Branch Pathway
    branch_output = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)

    # Combine pathways
    combined_output = layers.Add()([concatenated, branch_output])

    # Flatten and classify
    flatten = layers.Flatten()(combined_output)
    output = layers.Dense(units=10, activation='softmax')(flatten)

    model = models.Model(inputs=input_tensor, outputs=output)
    
    return model