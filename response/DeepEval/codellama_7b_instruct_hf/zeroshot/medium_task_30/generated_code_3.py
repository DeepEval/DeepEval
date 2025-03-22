import keras
from keras.models import Model
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dropout, Dense

# Define the input shape
input_shape = (32, 32, 3)

# Define the first pooling layer
pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_shape)

# Define the second pooling layer
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(pool1)

# Define the third pooling layer
pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(pool2)

# Flatten the outputs of the pooling layers
pool1_flat = Flatten()(pool1)
pool2_flat = Flatten()(pool2)
pool3_flat = Flatten()(pool3)

# Concatenate the flattened outputs
concat = Concatenate()([pool1_flat, pool2_flat, pool3_flat])

# Flatten the concatenated output
concat_flat = Flatten()(concat)

# Add two fully connected layers
fc1 = Dense(64, activation='relu')(concat_flat)
fc2 = Dense(10, activation='softmax')(fc1)

# Define the model
model = Model(inputs=input_shape, outputs=fc2)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Return the constructed model
return model