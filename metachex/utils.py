import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold

colormap =  lambda x, N: np.array(matplotlib.cm.get_cmap('viridis')(x/N))

def process_tSNE(features, learning_rate=10, perplexity=20):
    """ Computes tNSE embedding as array"""
    tsne = manifold.TSNE(n_components=2, init="random", learning_rate=learning_rate, random_state=0, perplexity=perplexity)
    encoded = tsne.fit_transform(features)
    return encoded

def plot_tsne(tsne_features, tsne_labels_one_hot, 
                              lables_names=None, 
                              num_subsample=None, 
                              visualize_class_list=None, 
                              plot_title='test', 
                              save_path='test.png'):

    tsne_labels = np.argmax(tsne_labels_one_hot, axis=-1)
    num_classes = tsne_labels_one_hot.shape[-1]
    num_samples = tsne_labels_one_hot.shape[0]

    if num_subsample is not None:
        # TODO:Visualize subset of samples
        pass

    if visualize_class_list is not None:
        # TODO: Visualize subset of classes
        pass

    if lables_names is None:
        plt.figure()
        plt.scatter(tsne_features[...,0], tsne_features[...,1], c=tsne_labels, cmap=plt.cm.get_cmap('viridis', num_classes))
        plt.title(plot_title)
        plt.colorbar(ticks=np.arange(num_classes))
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.figure()
        plt.title(plot_title)
        ax = plt.subplot(111)
        for i in range(num_classes):            
            scat = ax.scatter(
                tsne_features[tsne_labels==i,0], tsne_features[tsne_labels==i,1], 
                c=np.repeat(colormap(i, num_classes).reshape(-1,1), [tsne_features[tsne_labels==i,0].shape[0]]).reshape(4,-1).T,
                alpha=0.9,
                label=lables_names[i])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)

        plt.savefig(save_path)
    return

