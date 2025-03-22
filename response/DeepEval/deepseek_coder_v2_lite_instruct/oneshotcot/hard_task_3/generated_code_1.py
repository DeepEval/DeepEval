import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group
    processed_groups = []
    for group in split_layer:
        # 1x1 convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group)
        # 3x3 convolution
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        # Dropout for feature selection
        dropout = Dropout(0.25)(conv3x3)
        processed_groups.append(dropout)
    
    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(processed_groups)
    
    # Parallel branch pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Addition operation to combine the main pathway and branch pathway
    combined_pathway = tf.add(main_pathway, branch_pathway)
    
    # Flatten the result
    flatten_layer = Flatten()(combined_pathway)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()