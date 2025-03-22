from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, add

def dl_model():
    # Input layer for CIFAR-10 dataset
    input_img = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    pooled_output = GlobalAveragePooling2D()(x)
    fc1 = Dense(input_img.shape[3], activation='relu')(pooled_output)
    fc2 = Dense(input_img.shape[3], activation='sigmoid')(fc1)
    weights = Reshape((input_img.shape[3],))(fc2)
    weighted_x = multiply([x, weights])

    # Branch path
    branch_path = Conv2D(input_img.shape[3], (3, 3), padding='same')(input_img)

    # Combine both paths
    combined_output = add([weighted_x, branch_path])

    # Output layer
    fc3 = Dense(10, activation='softmax')(combined_output)

    # Create the model
    model = Model(inputs=input_img, outputs=fc3)

    return model

# Print the model summary
model = dl_model()
model.summary()