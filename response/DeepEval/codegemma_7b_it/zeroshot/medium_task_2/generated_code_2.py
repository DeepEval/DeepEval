from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dense, Flatten

def dl_model():

  # Define input layer
  inputs = Input(shape=(32, 32, 3), name='image_input')

  # Main path
  main_branch = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  main_branch = Conv2D(32, (3, 3), activation='relu')(main_branch)
  main_branch = MaxPooling2D(pool_size=(2, 2))(main_branch)

  # Branch path
  branch_path = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)

  # Combine features
  combined_features = concatenate([main_branch, branch_path])

  # Flatten and map to probabilities
  flatten = Flatten()(combined_features)
  output = Dense(10, activation='softmax', name='output')(flatten)

  # Create model
  model = Model(inputs=inputs, outputs=output)

  return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])