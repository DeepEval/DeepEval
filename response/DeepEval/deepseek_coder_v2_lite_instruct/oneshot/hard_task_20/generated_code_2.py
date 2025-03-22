import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_layer)

    # Split the input into three groups
    split1 = Lambda(lambda x: x[:, :16, :16, :])(input_layer)
    split2 = Lambda(lambda x: x[:, 16:, :16, :])(input_layer)
    split3 = Lambda(lambda x: x[:, :16, 16:, :])(input_layer)

    # Process each group with different kernels
    output1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1)
    output2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2)
    output3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split3)

    # Concatenate the outputs of the three groups
    main_path_output = Concatenate()([output1, output2, output3])

    # Branch Path
    branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fuse features by addition
    fused_features = tf.add(main_path_output, branch)

    # Flatten the fused features
    flatten_layer = Flatten()(fused_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()