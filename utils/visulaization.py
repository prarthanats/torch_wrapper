import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics(exp_metrics):
    sns.set(font_scale=1)
    fig, axs = plt.subplots(2,2,figsize=(25,15))
    plt.rcParams["figure.figsize"] = (25,6)
    train_accuracy,train_losses,test_accuracy,test_losses  = exp_metrics
    
    
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].set_title("Test Accuracy")

    axs[0, 0].plot(train_losses, label="Training Loss")
    axs[0,0].set_xlabel('epochs')
    axs[0,0].set_ylabel('loss')

    axs[1, 0].plot(train_accuracy, label="Training Accuracy")
    axs[1,0].set_xlabel('epochs')
    axs[1,0].set_ylabel('accuracy')

    axs[0, 1].plot(test_losses, label="Validation Loss")
    axs[0,1].set_xlabel('epochs')
    axs[0,1].set_ylabel('loss')

    axs[1, 1].plot(test_accuracy, label="Validation Accuracy")
    axs[1,1].set_xlabel('epochs')
    axs[1,1].set_ylabel('accuracy')
        
