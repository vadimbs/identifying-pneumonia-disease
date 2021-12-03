import os, shutil, errno, sys
from pathlib import Path
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.applications import VGG19

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix

import datetime
import pickle
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()



def list_files(path):
    """
    Print out directories name and files count

    Parameters
    ----------
        path: str
            Directory where to start from
    """

    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        
        files_count='  {} files'.format(len(files)) if len(files) is not 0 else ''
        print('{}{}/{}'.format(indent, os.path.basename(root), files_count))
        subindent = '\t' * (level + 1)
        #for f in files:
         #   print('{}{}'.format(subindent, f))

            
def copy_data(src, dst):
    """
    Copy data from 'src' folder to 'dst' folder recursively
    """
    
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise #None - 

            
            
def plot_acc_loss(history):
    """
    Plot model's train history
    """
    
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    fig.suptitle('Training and validation', fontsize=16)
    
    ax1.plot(epochs, acc, 'bo-', label='Training acc')
    ax1.plot(epochs, val_acc, 'ro-', label='Validation acc')
    ax1.set_title('Accuracy', fontsize=14)
    ax1.legend()
    
    ax2.plot(epochs, loss, 'bo-', label='Training loss')
    ax2.plot(epochs, val_loss, 'ro-', label='Validation loss')
    ax2.set_title('Loss', fontsize=14)
    ax2.legend()
    plt.show()


def plot_roc_pr_curves(y_test, y_hat_test):
    """
    Plot model's Receiver operating characteristic (ROC) and Precision Recall (PR) curves
    """
    
    fpr, tpr, tresholds = roc_curve(y_test, y_hat_test)
    roc_auc = roc_auc_score(y_test, y_hat_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_hat_test)
    pr_auc = auc(y_test, y_hat_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    fig.suptitle('ROC and PR curves', fontsize=16)
    
    ax1.plot(fpr,tpr, label = 'ROC AUC score: %.3f' % roc_auc)
    ax1.set_title('ROC', fontsize=14)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    
    ax2.plot(precision, recall, label = 'PR AUC score: %.3f' % pr_auc)
    ax2.set_title('PR', fontsize=14)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.legend()
    
    plt.show()


def plot_confusion_matrix(cm, class_names_list):
    """
    Plot Confusion matrix

    Parameters
    ----------
        cm: list
            Confusion matrix list
        class_names_list: list
    """
    
    cm_df = pd.DataFrame(cm, index = class_names_list, columns = class_names_list)
    figure = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix', fontsize=16)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


def print_predictive_values(cm):
    """
    Print out Positive and Negative predictive Value

    Parameters
    ----------
        cm: list
            Confusion matrix list
    """

    # Positive predictive Value = TP / (TP + FP)*100
    ppv = round(cm[0][0] / (cm[0][0] + cm[0][1]) * 100, 2)
    
    # Negative predictive Value = TN / (FN + TN)*100
    npv = round(cm[1][1] / (cm[1][0] + cm[1][1]) * 100, 2)

    print('Positive predictive Value:', ppv)
    print('Negative predictive Value:', npv)

    
def print_tpr_tnr(cm):
    """
    Print out Sensitivity and Specificity

    Parameters
    ----------
        cm: list
            Confusion matrix list
    """
    
    # Sensitivity (tpr) = tp / (tp + fn)
    tpr = round(cm[0][0] / (cm[0][0] + cm[1][0]), 2)
    
    # Specificity (tnr) = tn / (fp + tn)
    tnr = round(cm[1][0] / (cm[0][1] + cm[1][1]), 2)
    
    print('Sensitivity:', tpr)
    print('Specificity:', tnr)
    
    

def create_image_generators(data_dir, img_size, batch_size, aug=False):
    """
        Create image generators for model training, validating, and testing

        Parameters
        ----------
            data_dir: str
                Directory with images
            img_size: list
            batch_size: int
            aug: boolean, default=False
                Use data augmentation if set to True
                
        Return
        ------
            train_generator: tensorflow.python.keras.preprocessing.image.ImageDataGenerator
            val_generator: tensorflow.python.keras.preprocessing.image.ImageDataGenerator
            test_generator: tensorflow.python.keras.preprocessing.image.ImageDataGenerator
    """
    
    train_dir = '{}train'.format(data_dir)
    validation_dir = '{}val/'.format(data_dir)
    test_dir = '{}test/'.format(data_dir)
    
    datagen = ImageDataGenerator(rescale=1./255) if aug is False else ImageDataGenerator(rescale=1./255,
                                                                           rotation_range=40,
                                                                           width_shift_range=0.2,
                                                                           height_shift_range=0.2,
                                                                           shear_range=0.2,
                                                                           zoom_range=0.2,
                                                                           horizontal_flip=True,
                                                                           fill_mode='nearest')

    print('\nTrain set:')
    train_generator = datagen.flow_from_directory(train_dir, 
                                                  target_size=img_size, 
                                                  batch_size= batch_size,
                                                  seed=42,
                                                  class_mode='binary') 

    print('\nValidation set:')
    val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_dir, 
                                                target_size=img_size, 
                                                batch_size=batch_size,
                                                seed=42,
                                                class_mode='binary')

    print('\nTest set:')
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, 
                                                                            target_size=img_size, 
                                                                            batch_size=batch_size,
                                                                            class_mode='binary',
                                                                            seed=42,
                                                                            shuffle=False)
    
#     sys.exit()
    print('\nGenerator class indices:\n',train_generator.class_indices,'\n')
    
    
    return train_generator, val_generator, test_generator


def save_model(name, model, history=None, path='models/'):
    """
        Save trained model in HDF5, and its history in pickle format

        Parameters
        ----------
            name: str
                Model name to save as
            history: list, default=None
                Model fit history list
            path: str, default='models/'
                Directory where to store model
    """
    
    # Create directory if not exist
    os.makedirs(path, exist_ok=True)
    
    # TODO: create counter to add number at the end of model name in order to stop rewriting existing models

    model_path = os.path.join( path, '{}.h5'.format(name) )
    model.save(model_path)
    
    if history is not None:
        history_path = os.path.join( path, '{}_history.pickle'.format(name) )
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
            
            
def load_model(path):
    """
        Load saved model, load its history if exist

        Parameters
        ----------
            path: str
                Saved model path
                
        Return
        ------
            model: tensorflow.python.keras.engine.sequential.Sequential
                Model
            history: list 
                Model fit history
    """
    
    name = str(path).split('.')[0]
    
    model = models.load_model(name + '.h5')
    print('{} model'.format(name),  '\033[92m', 'loaded')
    
    history_file = name + '_history.pickle'
    if os.path.exists(history_file):
        with open(history_file, "rb") as f:
            history = pickle.load(f)
        print('{} history'.format(name),  '\033[92m', 'loaded')
    else: 
        history = None
        print('\033[91m', 'No history file found.')
        
    return model, history


def show_saved_models(path='models/'):
    """
        Print out saved models list, load selected model

        Parameters
        ----------
            path: str, default='models/'
                Directory with saved models
    """
    
    paths = sorted(Path(path).glob('*.h5'), key=os.path.getmtime) #, reverse=True)
    
    for i,model in enumerate(paths):
        print( i, str(os.path.basename(model)).split('.')[0] )
    
    print('\nLeave field empty to exit')
    val = input("Enter model number to load: ")
    print('\n')
    
    if val is not '': return load_model(paths[int(val)])
    
    
    
def test_model(model, data_dir, img_size, batch_size, aug, model_metrics):
    """
        test model and visualize its results

        Parameters
        ----------
            model: tensorflow.python.keras.engine.sequential.Sequential
            data_dir: str
                Directory with images
            img_size: list
            batch_size: int
            aug: boolean, default=False
                Use data augmentation if set to True
            model_metrics: list
    """
    
    train_generator, val_generator, test_generator = create_image_generators(data_dir, img_size, batch_size, aug)
    
    test_steps = len(test_generator)# if max_steps > len(test_generator) else max_steps
    
    
    # [test_loss, test_acc, ...]
    test = model.evaluate(test_generator, steps=test_steps)
    y_hat_test = model.predict(test_generator, steps=test_steps)
    print('\nTest evaluation generate {} predictions'.format(len(y_hat_test)))
    print('test:', test)
    
    print('\nMetrics \n--------------------------')
    model_metrics_names = list(model_metrics)
    for i in range(len(model_metrics_names)):
        print(model_metrics_names[i], ': ', test[i+1])
    
    y_test = test_generator.classes
    
    # ROC and Precision Recall (PR) curves
    plot_roc_pr_curves(y_test, y_hat_test)

    class_names_list = list(test_generator.class_indices.keys())
    cm = confusion_matrix(y_test, np.round(y_hat_test))
    plot_confusion_matrix(cm, class_names_list)
    print_predictive_values(cm)
    print_tpr_tnr(cm)

    

def process_model(model_name, model, model_metrics, data_dir, img_size=(150,150), batch_size=10, epochs=3, lr=1e-4, max_steps=None, aug=False):
    """
        Define, compile, fit, test, save model and visualize its results

        Parameters
        ----------
            model_name: str
                Model name to save as
            model: tensorflow.python.keras.engine.sequential.Sequential
            model_metrics: list
            data_dir: str
                Directory with images
            img_size: list, default=(150,150)
            batch_size: int, default=10
            epochs: int default=3
            lr: float, default=1e-4
                Model learning rate
            max_steps: int, default=None
                Max steps per batch
            aug: boolean, default=False
                Use data augmentation if set to True
            
        Return
        ------
            model: tensorflow.python.keras.engine.sequential.Sequential
                Model
            history: list 
                Model fit history
    """

    # Reshape the data
    train_generator, val_generator, test_generator = create_image_generators(data_dir, img_size, batch_size, aug)

    
    train_steps = len(train_generator) if max_steps > len(train_generator) else max_steps
    val_steps = len(val_generator) if max_steps > len(val_generator) else max_steps
    

    # Fit the model
    start = datetime.datetime.now()
    
    history = model.fit(train_generator,
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=val_steps)

    end = datetime.datetime.now()
    elapsed = end - start
    print('\nTraining took a total of {}\n'.format(elapsed))
    
    
    # Save model and history
    model_name = '{}_{}x{}_{}e'.format(model_name, img_size[0], img_size[1], epochs)
    save_model(model_name, model, history)
    
    
    # Visualize the history
    # Accuracy loss
    plot_acc_loss(history.history)
    
    #if 'Accuracy' in model_metrics:
    
    # Test the model
    test_model(model, data_dir, img_size, batch_size, aug, model_metrics)
    
    
    return model, history