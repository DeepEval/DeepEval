import keras
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Dense, Flatten
from keras.models import Model

# Define the input shape
input_shape = (32, 32, 3)

# Define the first block
def block_1(input_tensor):
    # Split the input into three groups using tf.split
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
    # Extract features using separable convolutional layers with different kernel sizes
    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    # Concatenate the outputs of the three groups
    output_tensor = Concatenate()([conv1, conv2, conv3])
    return output_tensor

# Define the second block
def block_2(input_tensor):
    # Define the four parallel branches
    path_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path_3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path_4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    # Concatenate the outputs of the four branches
    output_tensor = Concatenate()([path_1, path_2, path_3, path_4])
    return output_tensor

# Define the input layer
input_layer = Input(shape=input_shape)

# Apply the first block
output_tensor = block_1(input_tensor=input_layer)

# Apply the second block
output_tensor = block_2(input_tensor=output_tensor)

# Flatten the output and apply a fully connected layer
flatten_layer = Flatten()(output_tensor)
output_layer = Dense(units=10, activation='softmax')(flatten_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])