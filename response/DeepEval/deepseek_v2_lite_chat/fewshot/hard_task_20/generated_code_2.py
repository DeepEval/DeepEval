import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_1[1])
    conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_1[2])

    # Branch path
    split_2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_2[0])
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_2[1])
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_2[2])

    # Concatenate and add
    concat = Concatenate()([conv1_3, conv2_3])
    add = Add()([concat, conv1_2, conv2_2])

    # Fully connected layers
    flatten = Flatten()(add)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()