import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Process each group with convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[2])
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split2[0])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split2[1])
    conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split2[2])
    
    conv3_1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(split3[0])
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(split3[1])
    conv3_3 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(split3[2])
    
    # Pooling along with dropout
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    
    dropout1 = keras.layers.Dropout(0.2)(pool1)
    dropout2 = keras.layers.Dropout(0.2)(pool2)
    dropout3 = keras.layers.Dropout(0.2)(pool3)
    
    # Concatenate the outputs from each group
    concat = Concatenate()([dropout1, dropout2, dropout3])
    
    # Branch pathway
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Add main and branch pathways
    combined = Add()([dropout1, conv4_1])
    
    # Fully connected layer
    dense = Dense(units=256, activation='relu')(combined)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])