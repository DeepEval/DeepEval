import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv_1x1 = Conv2D(32, (1, 1), activation='relu')(split_channels[0])
    conv_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])

    dropout_1 = Dropout(0.2)(conv_1x1)
    dropout_2 = Dropout(0.2)(conv_3x3)
    dropout_3 = Dropout(0.2)(conv_5x5)

    concatenated = Concatenate()([dropout_1, dropout_2, dropout_3])

    # Block 2
    branch_1 = Conv2D(64, (1, 1), activation='relu')(concatenated)

    branch_2_conv1 = Conv2D(64, (1, 1), activation='relu')(concatenated)
    branch_2_conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch_2_conv1)

    branch_3_conv1 = Conv2D(64, (1, 1), activation='relu')(concatenated)
    branch_3_conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch_3_conv1)

    branch_4_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concatenated)
    branch_4_conv = Conv2D(64, (1, 1), activation='relu')(branch_4_pool)

    concatenated_2 = Concatenate()([branch_1, branch_2_conv2, branch_3_conv2, branch_4_conv])

    # Output layer
    flattened = Flatten()(concatenated_2)
    output_layer = Dense(10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of compiling the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])