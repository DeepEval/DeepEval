import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, MaxPooling2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    # Block 1: Main Path
    main_path = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Block 1: Branch Path
    branch_path = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Block 2: Feature Extraction
    feature_extraction = Sequential([
        Lambda(lambda x: tf.split(x, 3, axis=-1)),
        SeparableConv2D(16, (1, 1), activation='relu'),
        Dropout(0.2),
        SeparableConv2D(32, (3, 3), activation='relu'),
        Dropout(0.2),
        SeparableConv2D(64, (5, 5), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Model
    model = Model(inputs=main_path.input, outputs=main_path(branch_path(feature_extraction(input_shape))))

    return model