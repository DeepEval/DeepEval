# Import necessary packages
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)  # Split input along channel axis
    block1_x = []
    for i in range(3):
        x_i = Conv2D(32, (1, 1), activation='relu', name='block1_conv_' + str(i))(x[i])
        block1_x.append(x_i)

    # Apply dropout to reduce overfitting
    block1_x = [Dropout(0.2)(x_i) for x_i in block1_x]

    # Concatenate the outputs from the three groups
    block1_x = Concatenate(axis=-1)(block1_x)
    block1_x = layers.Reshape((-1,))(block1_x)  # Flatten block1_x

    # Block 2
    x = Lambda(lambda x: tf.split(x, 4, axis=-1))(block1_x)  # Split block1_x along channel axis
    block2_x = []
    block2_x.append(Conv2D(64, (1, 1), activation='relu', name='block2_conv_0')(x[0]))
    block2_x.append(Conv2D(64, (1, 1), activation='relu', name='block2_conv_1')(x[1]))
    block2_x.append(Conv2D(64, (3, 3), activation='relu', name='block2_conv_2')(x[2]))
    block2_x.append(Conv2D(64, (5, 5), activation='relu', name='block2_conv_3')(x[3]))
    block2_x.append(MaxPooling2D((3, 3), strides=(2, 2), name='block2_max_pool')(x[2]))
    block2_x.append(Conv2D(64, (1, 1), activation='relu', name='block2_conv_4')(block2_x[-1]))

    # Concatenate the outputs from all branches
    block2_x = Concatenate(axis=-1)(block2_x)
    block2_x = layers.Reshape((-1,))(block2_x)  # Flatten block2_x

    # Output layer
    x = Dense(10, activation='softmax', name='output_layer')(block2_x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Compile the model (optional)
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model