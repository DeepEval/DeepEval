from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    main_path = Conv2D(64, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Add()([main_path, branch_path])

    # Block 2: Split the input into three groups along the channel
    input_layer = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)

    # Block 2.1: Extract features using depthwise separable convolutional layers with different kernel sizes
    features1 = Conv2D(32, (1, 1), activation='relu')(input_layer[0])
    features1 = Conv2D(32, (3, 3), activation='relu')(features1)
    features2 = Conv2D(32, (3, 3), activation='relu')(input_layer[1])
    features2 = Conv2D(32, (5, 5), activation='relu')(features2)
    features3 = Conv2D(32, (5, 5), activation='relu')(input_layer[2])
    features3 = Conv2D(32, (7, 7), activation='relu')(features3)

    # Block 2.2: Concatenate the outputs from the three groups
    features = Concatenate()([features1, features2, features3])

    # Block 3: Fully connected layers for classification
    features = GlobalAveragePooling2D()(features)
    features = Dense(128, activation='relu')(features)
    features = Dense(10, activation='softmax')(features)

    # Define the model
    model = Model(inputs=input_layer, outputs=features)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model