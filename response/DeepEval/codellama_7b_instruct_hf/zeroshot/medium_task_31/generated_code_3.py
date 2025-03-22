from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

# Define the input layer
input_layer = Input(shape=(32, 32, 3))

# Define the first Lambda layer
split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

# Define the first Conv2D layer
conv1_layer = Conv2D(32, (1, 1), activation='relu')(split_layer[0])

# Define the second Conv2D layer
conv2_layer = Conv2D(64, (3, 3), activation='relu')(split_layer[1])

# Define the third Conv2D layer
conv3_layer = Conv2D(128, (5, 5), activation='relu')(split_layer[2])

# Define the concatenation layer
concat_layer = Concatenate()([conv1_layer, conv2_layer, conv3_layer])

# Define the MaxPooling2D layer
pooling_layer = MaxPooling2D((2, 2))(concat_layer)

# Define the Flatten layer
flatten_layer = Flatten()(pooling_layer)

# Define the first Dense layer
dense1_layer = Dense(128, activation='relu')(flatten_layer)

# Define the second Dense layer
dense2_layer = Dense(10, activation='softmax')(dense1_layer)

# Define the output layer
output_layer = Dense(10, activation='softmax')(dense2_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])