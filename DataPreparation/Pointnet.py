import glob
import open3d as o3d
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import random
from tensorflow.keras.callbacks import ModelCheckpoint
tf.random.set_seed(1234)

def load_dataset(num_points=3000):
    
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    #Read all training TOF point clouds
    path = "AugmentedData/TOF/Train"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            #pcd = np.asarray(pcd.points)[:num_points]
            pcd = random.choices(np.asarray(pcd.points,dtype='float32').reshape(-1,3),k=num_points)
            train_points.append(pcd)
            train_labels.append(0)

    #Read all training PC point clouds
    path = "AugmentedData/PC/Train"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            #pcd = np.asarray(pcd.points)[:num_points]
            pcd = random.choices(np.asarray(pcd.points,dtype='float32').reshape(-1,3),k=num_points)
            train_points.append(pcd)
            train_labels.append(1)
    
    #Read all test TOF point clouds
    path = "AugmentedData/TOF/Test"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            #pcd = np.asarray(pcd.points)[:num_points]
            pcd = random.choices(np.asarray(pcd.points,dtype='float32').reshape(-1,3),k=num_points)
            test_points.append(pcd)
            test_labels.append(0)

    #Read all test PC point clouds
    path = "AugmentedData/PC/Test"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            #pcd = np.asarray(pcd.points)[:num_points]
            pcd = random.choices(np.asarray(pcd.points,dtype='float32').reshape(-1,3),k=num_points)
            test_points.append(pcd)
            test_labels.append(1)

    return train_points, train_labels, test_points, test_labels

def train_test(num_points, batch_size, crop=False):
    NUM_POINTS = num_points
    NUM_CLASSES = 2
    BATCH_SIZE = batch_size

    train_points, train_labels, test_points, test_labels = load_dataset(NUM_POINTS)


    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    train_dataset = train_dataset.shuffle(len(train_points)).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)


    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
        
    def tnet(inputs, num_features):

        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])


    inputs = keras.Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
    checkpoint_path = "PointnetCheckpoint/pointnet.h5"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       monitor='val_loss', 
                                       mode='min', 
                                       verbose=1)

    if crop:
        # Define the L2 regularization parameter
        l2 = tf.keras.regularizers.l2(0.01)
        optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        optimizer.kernel_regularizer = l2
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["sparse_categorical_accuracy"],
        )

        model.fit(train_dataset, epochs=30, validation_data=test_dataset, callbacks=[checkpoint_callback])
    else:
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["sparse_categorical_accuracy"],
        )

        model.fit(train_dataset, epochs=20, validation_data=test_dataset, callbacks=[checkpoint_callback])

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # Load the saved model
    model.load_weights('PointnetCheckpoint/pointnet.h5')

    paths = []
    preds = []
    argmax = []
    correct = []
    #Read all test TOF point clouds
    path = "AugmentedData/TOF/Test"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            pcd = np.asarray(pcd.points)[:NUM_POINTS]
            #pcd = np.asarray(random.choices(pcd,k=NUM_POINTS))
            pcd = np.expand_dims(pcd, axis=0)
            prediction = model.predict(pcd)
            preds.append(prediction)
            paths.append(str(f))
            agmax = tf.math.argmax(prediction, -1)[0].numpy()
            if agmax == 0:
                correct.append("Yes")
            else: correct.append("No")
            argmax.append(prediction[0][agmax])

    #Read all test PC point clouds
    path = "AugmentedData/PC/Test"+"/*.pcd"
    for f in glob.glob(path):
        pcd = o3d.io.read_point_cloud(str(f))
        if pcd.is_empty(): 
            exit()
        else: 
            pcd = np.asarray(pcd.points)[:NUM_POINTS]
            #pcd = np.asarray(random.choices(pcd,k=NUM_POINTS))
            pcd = np.expand_dims(pcd, axis=0)
            prediction = model.predict(pcd)
            preds.append(prediction)
            paths.append(str(f))
            agmax = tf.math.argmax(prediction, -1)[0].numpy()
            if agmax == 1:
                correct.append("Yes")
            else: correct.append("No")
            argmax.append(prediction[0][agmax])


    df = pd.DataFrame({'Path': paths, 'Prediction': preds,'Argmax': argmax, 'Correct': correct})
    df = df.sort_values(by='Argmax',ascending=True)
    df.to_excel('Xlsx/PointNet.xlsx', index=False)

