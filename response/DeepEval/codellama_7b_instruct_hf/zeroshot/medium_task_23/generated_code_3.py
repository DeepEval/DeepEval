from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Define the input layer
input_layer = Input(shape=(32, 32, 3))

# Define the first branch
conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

# Define the second branch
conv2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
conv2 = Conv2D(32, (1, 7), activation='relu')(conv2)
conv2 = Conv2D(32, (7, 1), activation='relu')(conv2)

# Define the third branch
conv3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
conv3 = Conv2D(32, (1, 7), activation='relu')(conv3)
conv3 = Conv2D(32, (7, 1), activation='relu')(conv3)
conv3 = Conv2D(32, (1, 7), activation='relu')(conv3)
conv3 = Conv2D(32, (7, 1), activation='relu')(conv3)

# Define the fourth branch
pooling = MaxPooling2D((2, 2))(input_layer)
pooling = Conv2D(32, (1, 1), activation='relu')(pooling)

# Define the fused outputs
fused_outputs = concatenate([conv1, conv2, conv3, pooling], axis=1)

# Define the final layer
x = Flatten()(fused_outputs)
x = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=input_layer, outputs=x)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))