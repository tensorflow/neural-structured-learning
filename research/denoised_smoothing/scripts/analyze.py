# Code courtesy: https://github.com/microsoft/denoised-smoothing/blob/master/code/analyze.py

from easydict import EasyDict as edict
from typing import *
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import seaborn as sns
sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def get_abstention_rate(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return 1.*(df["predict"]==-1).sum()/len(df["predict"])*100

class ApproximateAccuracy_API(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, header=None, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df[df.columns[1]] & (df[df.columns[2]] >= radius)).mean()

    def get_abstention_rate(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return 1.*(df[df.columns[-1]]==-1).sum()/len(df[df.columns[-1]])*100

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str = None, plot_fmt: str = "", scale_x: float = 1, alpha: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.alpha = alpha



def plot_certified_accuracy_per_sigma_against_baseline(outfile: str, title: str, max_radius: float,
                            methods: List[Line]=None, label='Ours', methods_base: List[Line]=None, label_base='Baseline', radius_step: float = 0.01, upper_bounds=False) -> None:
    color = ['b', 'orange', 'g', 'r']

    sigmas = [0.12, 0.25, 0.5, 1.00]
    if "api" in outfile:
        sigmas = [0.12, 0.25]

    for it, sigma in enumerate(sigmas):
        methods_sigma = [method for method in methods if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_sigma, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), color[it], label='{}|$\sigma = {:.2f}$'.format(label, sigma))

    for it, line in enumerate(methods_base):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], dashes=[2, 2], alpha=line.alpha, label='{}|'.format(label_base)+line.legend)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    if "api" not in outfile:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_per_sigma_against_baseline_finetune(outfile: str, title: str, max_radius: float,
                            methods: List[Line]=None, label='Ours', methods_finetune=None, label_finetune="Finetune", methods_base: List[Line]=None, label_base='Baseline', radius_step: float = 0.01, upper_bounds=False) -> None:
    color = ['b', 'orange', 'g', 'r']

    sigmas = [0.12, 0.25, 0.5, 1.00]
    if "api" in outfile:
        sigmas = [0.25]

    for it, sigma in enumerate(sigmas):
        methods_eps = [method for method in methods_finetune if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_eps, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), color[3], label='{}|$\sigma = {:.2f}$'.format(label_finetune, sigma))

    for it, sigma in enumerate(sigmas):
        methods_eps = [method for method in methods if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_eps, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), color[0], label='{}|$\sigma = {:.2f}$'.format(label, sigma))

    for it, line in enumerate(methods_base):
        if "0.25" not in line.quantity.data_file_path:
            continue
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[1],             alpha=line.alpha, label='{}|'.format(label_base)+line.legend)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    if "api" not in outfile:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_per_sigma_best_model(outfile: str, title: str, max_radius: float,
                            methods: List[Line]=None, label='Ours', methods_base: List[Line]=None, label_base='Baseline', radius_step: float = 0.01, upper_bounds=False, sigmas=[0.25]) -> None:
    color = ['b', 'orange', 'g', 'r']
    for it, sigma in enumerate(sigmas):
        methods_sigma = [method for method in methods if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_sigma, 0, max_radius, radius_step)
        accuracies_cert_ours = np.nan_to_num(accuracies_cert_ours, -1)
        plt.plot(radii, accuracies_cert_ours[accuracies_cert_ours[:,0].argmax(), :], color[it], label='{}|$\sigma = {:.2f}$'.format(label, sigma))
    for it, sigma in enumerate(sigmas):
        methods_sigma_base = [method for method in methods_base if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_sigma_base, 0, max_radius, radius_step)
        accuracies_cert_ours = np.nan_to_num(accuracies_cert_ours, -1)
        plt.plot(radii, accuracies_cert_ours[accuracies_cert_ours[:,0].argmax(), :], color[it], dashes=[2, 2], label='{}|$\sigma = {:.2f}$'.format(label_base, sigma))

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_one_sigma_best_model_multiple_methods(outfile: str, title: str, max_radius: float,
                            methods_labels_colors_dashes: List,
                            radius_step: float = 0.01, upper_bounds=False, sigma=0.25) -> None:
    for it, (methods, label, color, dashes) in enumerate(methods_labels_colors_dashes):
        methods_sigma = [method for method in methods if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_sigma, 0, max_radius, radius_step)
        accuracies_cert_ours = np.nan_to_num(accuracies_cert_ours, -1)
        plt.plot(radii, accuracies_cert_ours[accuracies_cert_ours[:,0].argmax(), :], 
                    color, dashes=dashes, linewidth=2, label=label)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def latex_table_certified_accuracy_upper_envelope(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]=None, clean_accuracy=True):
    accuracies, radii = _get_accuracies_at_radii(methods, radius_start, radius_stop, radius_step)
    clean_accuracies, _ = _get_accuracies_at_radii(methods, 0, 0, 0.25)
    assert clean_accuracies.shape[1] == 1

    f = open(outfile, 'w')

    f.write("$\ell_2$ Radius")
    for radius in radii:
        f.write("& ${:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    clean_accuracies = np.nan_to_num(clean_accuracies, -1)
    accuracies = np.nan_to_num(accuracies, -1)
    for j, radius in enumerate(radii):
        argmaxs = np.argwhere(accuracies[:,j] == accuracies[:, j].max())
        argmaxs = argmaxs.flatten()
        i = argmaxs[clean_accuracies[argmaxs, 0].argmax()]
        # i = i.flatten()[0]
        if clean_accuracy:
            txt = " & $^{("+"{:.2f})".format(clean_accuracies[i, 0]) + "}" + "${:.2f}".format(accuracies[i, j])
        else:
            txt = " & {:.2f}".format(accuracies[i, j])
        f.write(txt)
    f.write("\\\\\n")
    f.close()

def _get_accuracies_at_radii(methods: List[Line], radius_start: float, radius_stop: float, radius_step: float):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)
    return accuracies, radii


if __name__ == "__main__":
    if not os.path.isdir("analysis/plots/cifar10/full_access"):
        os.makedirs("analysis/plots/cifar10/full_access")
    if not os.path.isdir("analysis/plots/cifar10/query_access"):
        os.makedirs("analysis/plots/cifar10/query_access")

    if not os.path.isdir("analysis/plots/imagenet/full_access"):
        os.makedirs("analysis/plots/imagenet/full_access")
    if not os.path.isdir("analysis/plots/imagenet/query_access"):
        os.makedirs("analysis/plots/imagenet/query_access")

    if not os.path.isdir("analysis/plots/vision_api/azure/"):
        os.makedirs("analysis/plots/vision_api/azure/")
    if not os.path.isdir("analysis/plots/vision_api/google/"):
        os.makedirs("analysis/plots/vision_api/google/")
    if not os.path.isdir("analysis/plots/vision_api/aws/"):
        os.makedirs("analysis/plots/vision_api/aws/")
    if not os.path.isdir("analysis/plots/vision_api/clarifai/"):
        os.makedirs("analysis/plots/vision_api/clarifai/")

    if not os.path.isdir("analysis/latex/"):
        os.makedirs("analysis/latex/")

################### PLOTS
# Paper plots
    all_cifar_cohen_N10000=[
            Line(ApproximateAccuracy("data/certify/cifar10/no_denoiser/MODEL_resnet110_90epochs/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.5, 1.0]
        ]
    cifar_no_denoiser_N10000 = [
            Line(ApproximateAccuracy("data/certify/cifar10/no_denoiser/MODEL_resnet110_90epochs/noise_0.00/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.5, 1.0]
        ]
    cifar_denoiser_cifar10_dncnn_epochs_90_N10000 = [
            Line(ApproximateAccuracy("data/certify/cifar10/mse_obj/MODEL_resnet110_90epochs_DENOISER_cifar10_dncnn_epochs_90/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.5, 1.0]
        ]
    cifar_denoiser_cifar10_dncnn_wide_epochs_90_N10000 = [
            Line(ApproximateAccuracy("data/certify/cifar10/mse_obj/MODEL_resnet110_90epochs_DENOISER_cifar10_dncnn_wide_epochs_90/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.5, 1.0]
        ]
    cifar_denoiser_cifar10_memnet_epochs_90_N10000 = [
            Line(ApproximateAccuracy("data/certify/cifar10/mse_obj/MODEL_resnet110_90epochs_DENOISER_cifar10_memnet_epochs_90/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.5, 1.0]
        ]

    all_cifar_denoising_obj_denoisers =   cifar_denoiser_cifar10_dncnn_epochs_90_N10000 + \
                                    cifar_denoiser_cifar10_dncnn_wide_epochs_90_N10000 + \
                                    cifar_denoiser_cifar10_memnet_epochs_90_N10000

### Full-Access Cifar10
    denoiser_networks = ['dncnn', 'dncnn_wide', 'memnet']
    all_exp_resnet110_fullAccess_cifar10_classification = [
        Line(ApproximateAccuracy("data/certify/cifar10/clf_obj/{0}_resnet110_90epochs_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        # each of these correspond to a hyperparamter setting (see the appendix in the paper for details)
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_1',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_2',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_3',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_4',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_5',
        ] 
        for denoiser in denoiser_networks 
        ]
    all_exp_resnet110_fullAccess_cifar10_stability = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab_obj/{0}_resnet110_90epochs_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        # each of these correspond to a hyperparamter setting (see the appendix in the paper for details)
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_1',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_2',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_3',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_4',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_5',
        ] 
        for denoiser in denoiser_networks 
        ]
    all_exp_resnet110_fullAccess_cifar10_stability_finetune = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab+mse_obj/{0}_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-4_20epochs_renset110_90epochs',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-5_20epochs_renset110_90epochs',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-4_20epochs_renset110_90epochs',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-5_20epochs_renset110_90epochs',
        ] 
        for denoiser in denoiser_networks 
        ]

    # Plot best models
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/full_access/resnet110_90epochs_all_methods_sigma_12", 'Query-access Cifar10-ResNet110', 1.0,
        methods_labels_colors_dashes=[ 
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_fullAccess_cifar10_stability, 'Stab', 'g', [6, 2]),
            (all_exp_resnet110_fullAccess_cifar10_stability_finetune, 'Stab+MSE', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.12)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/full_access/resnet110_90epochs_all_methods_sigma_25", 'Query-access Cifar10-ResNet110', 1.0,
        methods_labels_colors_dashes=[ 
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_fullAccess_cifar10_stability, 'Stab', 'g', [6, 2]),
            (all_exp_resnet110_fullAccess_cifar10_stability_finetune, 'Stab+MSE', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.25)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/full_access/resnet110_90epochs_all_methods_sigma_50", 'Query-access Cifar10-ResNet110', 1.0,
        methods_labels_colors_dashes=[ 
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_fullAccess_cifar10_stability, 'Stab', 'g', [6, 2]),
            (all_exp_resnet110_fullAccess_cifar10_stability_finetune, 'Stab+MSE', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.50)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/full_access/resnet110_90epochs_all_methods_sigma_100", 'Query-access Cifar10-ResNet110', 1.0,
        methods_labels_colors_dashes=[ 
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_fullAccess_cifar10_stability, 'Stab', 'g', [6, 2]),
            (all_exp_resnet110_fullAccess_cifar10_stability_finetune, 'Stab+MSE', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=1.00)

    plot_certified_accuracy_per_sigma_best_model(
        "analysis/plots/cifar10/full_access/resnet110_90epochs_stab_vs_clf", 'Stability vs. Classification', 2.25,
        methods=all_exp_resnet110_fullAccess_cifar10_stability, label='Stab',
        methods_base=all_exp_resnet110_fullAccess_cifar10_classification, label_base='Clf',
        sigmas=[0.12, 0.25, 0.5, 1.0])

#######################################################################################
### Query-Access Cifar10
    denoiser_networks = ['dncnn', 'dncnn_wide', 'memnet']
    all_exp_resnet110_queryAccess_cifar10_classification = [
        Line(ApproximateAccuracy("data/certify/cifar10/clf_obj/{0}_multi_classifiers_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        # each of these correspond to a hyperparamter setting (see the appendix in the paper for details)
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_1',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_2',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_3',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_4',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_5',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_classification_obj_adamThenSgd_6',
        ] 
        for denoiser in denoiser_networks 
        ]
    all_exp_resnet110_queryAccess_cifar10_stability = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab_obj/{0}_multi_classifiers_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        # each of these correspond to a hyperparamter setting (see the appendix in the paper for details)
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_1',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_2',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_3',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_4',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_5',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_smoothness_obj_adamThenSgd_6',
        ] 
        for denoiser in denoiser_networks 
        ]
    all_exp_resnet110_queryAccess_cifar10_stability_1surrogate = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab_obj/MODEL_ResNet110_DENOISER_surrogate_resnet110/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        ]
    all_exp_resnet110_queryAccess_cifar10_stability_finetune_1surrogate = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab+mse_obj/{0}_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-4_20epochs_WRN',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-5_20epochs_WRN',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-4_20epochs_WRN',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-5_20epochs_WRN',
        ] 
        for denoiser in denoiser_networks 
        ]
    all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate = [
        Line(ApproximateAccuracy("data/certify/cifar10/stab+mse_obj/{0}_{1}/noise_{2:.2f}/test_N10000/sigma_{2:.2f}".format(exp, denoiser, noise)), "$\sigma = {:.2f}$".format(noise))
        for noise in [0.12, 0.25, 0.50, 1.00] 
        for exp in [
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-4_20epochs_multi_classifiers',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_adam_1e-5_20epochs_multi_classifiers',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-4_20epochs_multi_classifiers',
        'MODEL_resnet110_90epochs_DENOISER_cifar10_finetune_smoothness_obj_sgd_1e-5_20epochs_multi_classifiers',
        ] 
        for denoiser in denoiser_networks 
        ]

    # Plot best models
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_12", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.12)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_25", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.25)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_50", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=0.50)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_100", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_cifar_cohen_N10000, 'White-box', 'b', [1, 0]),
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_cifar_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
            (cifar_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
        ],
        sigma=1.00)

    plot_certified_accuracy_per_sigma_best_model(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_stab_vs_clf", 'finetune_cifar_best_models', 2.25,
        methods=all_exp_resnet110_queryAccess_cifar10_stability, label='Stab',
        methods_base=all_exp_resnet110_queryAccess_cifar10_classification, label_base='Clf',
        sigmas=[0.12, 0.25, 0.5, 1.0])
    ## 1 Surrogate 
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_12_1surrogate_vs_14", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_1surrogate, 'Stab 1-Surrogate', 'b', [2, 4]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_1surrogate, 'Stab+MSE 1-Surrogate', 'k', [5, 1]),
        ],
        sigma=0.12)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_25_1surrogate_vs_14", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_1surrogate, 'Stab 1-Surrogate', 'b', [2, 4]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_1surrogate, 'Stab+MSE 1-Surrogate', 'k', [5, 1]),
        ],
        sigma=0.25)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_50_1surrogate_vs_14", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_1surrogate, 'Stab 1-Surrogate', 'b', [2, 4]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_1surrogate, 'Stab+MSE 1-Surrogate', 'k', [5, 1]),
        ],
        sigma=0.50)
    plot_certified_accuracy_one_sigma_best_model_multiple_methods(
        "analysis/plots/cifar10/query_access/resnet110_90epochs_all_methods_sigma_100_1surrogate_vs_14", 'blackbox_cifar_best_models', 1.0,
        methods_labels_colors_dashes=[
            (all_exp_resnet110_queryAccess_cifar10_stability, 'Stab 14-Surrogates', 'g', [6, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_1surrogate, 'Stab 1-Surrogate', 'b', [2, 4]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_14surrogate, 'Stab+MSE 14-Surrogates', 'orange', [4, 2]),
            (all_exp_resnet110_queryAccess_cifar10_stability_finetune_1surrogate, 'Stab+MSE 1-Surrogate', 'k', [5, 1]),
        ],
        sigma=1.00)

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
#################################################################################################################

    #Imagenet results
    imagenet_archs = ['resnet18', 'resnet34', 'resnet50']
    imagenet_results = edict()

    for arch in imagenet_archs:
        imagenet_results[arch] = edict()

        imagenet_results[arch].imagenet_no_denoiser_N10000 = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}/noise_0.00/test_N10000/sigma_{1:.2f}".format(arch, noise)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            ]
        imagenet_results[arch].imagenet_denoiser_dncnn_off_the_shelf_N10000 = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_dncnn-off-the-shelf/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            ]
        imagenet_results[arch].imagenet_denoiser_imagenet_dncnn_5epoch_lr1e_4_N10000 = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_imagenet_dncnn_5epoch_lr1e-4/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            ]

        imagenet_results[arch].all_imagenet_denoising_obj_denoisers = imagenet_results[arch].imagenet_denoiser_imagenet_dncnn_5epoch_lr1e_4_N10000
                                                    #    imagenet_results[arch].imagenet_denoiser_dncnn_off_the_shelf_N10000 + \
                                        

        ## Classification objective denoisers
        imagenet_results[arch].all_imagenet_classification_obj_N10000 = {}
        imagenet_results[arch].all_imagenet_classification_obj_N10000['resnet18'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet18/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_classification_obj_adam_1e-5_20epochs',
                            ]
        ]
        imagenet_results[arch].all_imagenet_classification_obj_N10000['resnet34'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet34/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_classification_obj_adam_1e-5_20epochs',
                            ]
        ]
        imagenet_results[arch].all_imagenet_classification_obj_N10000['resnet50'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet50/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_classification_obj_adam_1e-5_20epochs',
                            ]
        ]
        ## Stability Objective denoisers
        imagenet_results[arch].all_imagenet_stability_obj_N10000 = {}
        imagenet_results[arch].all_imagenet_stability_obj_N10000['resnet18'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet18/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_smoothness_obj_adam_1e-5_20epochs',
                            ]
        ]
        imagenet_results[arch].all_imagenet_stability_obj_N10000['resnet34'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet34/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_smoothness_obj_adam_1e-5_20epochs',
                            ]
        ]
        imagenet_results[arch].all_imagenet_stability_obj_N10000['resnet50'] = [
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{0}_DENOISER_{2}/resnet50/dncnn/noise_{1:.2f}/test_N10000/sigma_{1:.2f}".format(arch, noise, denoiser)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.12, 0.25, 0.5, 1.0]
            for denoiser in ['imagenet_smoothness_obj_adam_1e-5_20epochs',
                            ]
        ]

        imagenet_results[arch].cohen_training_N10000 = {}
        imagenet_results[arch].cohen_training_N10000=[
                Line(ApproximateAccuracy("data/certify/imagenet/MODEL_{1}/noise_{0:.2f}/test_N10000/sigma_{0:.2f}".format(noise, arch)), "$\sigma = {0:.2f}$".format(noise))
            for noise in [0.25, 0.5, 1.0]
            ]

    # Imagenet plots
    for arch in imagenet_archs:
        ### Full-Access
        plot_certified_accuracy_per_sigma_best_model(
            "analysis/plots/imagenet/full_access/MODEL_{}_Stab_vs_Clf".format(arch), '{} Stab vs Clf'.format(arch), 2.25, 
            methods=imagenet_results[arch].all_imagenet_stability_obj_N10000[arch], label='Stab+MSE',
            methods_base=imagenet_results[arch].all_imagenet_classification_obj_N10000[arch], label_base='Clf+MSE',
            sigmas=[0.12, 0.25, 0.5, 1.0])

        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/full_access/MODEL_{}_all_methods_stability_sigma_25".format(arch), 'fixed_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),
                (imagenet_results[arch].all_imagenet_stability_obj_N10000[arch], 'Stab+MSE', 'orange', [4, 2]),
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=0.25)
        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/full_access/MODEL_{}_all_methods_stability_sigma_50".format(arch), 'fixed_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),
                (imagenet_results[arch].all_imagenet_stability_obj_N10000[arch], 'Stab+MSE', 'orange', [4, 2]),
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=0.50)
        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/full_access/MODEL_{}_all_methods_stability_sigma_100".format(arch), 'fixed_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),
                (imagenet_results[arch].all_imagenet_stability_obj_N10000[arch], 'Stab+MSE', 'orange', [4, 2]),
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=1.00)


        ### Query-Access
        surrogate_models = [(imagenet_results[arch].all_imagenet_stability_obj_N10000[b], 'Stab+MSE-{}'.format(b), color, dashes) 
            for b, color, dashes in zip(set(imagenet_archs) - set([arch]), 
                                            ['g', 'orange'], 
                                            [[6, 2], [4, 2]], 
                                        )
                            ]
        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/query_access/MODEL_{}_stability_sigma_25_with_surrogate".format(arch), 'blackbox_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),] + surrogate_models + [
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=0.25)
        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/query_access/MODEL_{}_stability_sigma_50_with_surrogate".format(arch), 'blackbox_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),] + surrogate_models + [
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=0.50)
        plot_certified_accuracy_one_sigma_best_model_multiple_methods(
            "analysis/plots/imagenet/query_access/MODEL_{}_stability_sigma_100_with_surrogate".format(arch), 'blackbox_imagenet_best_models', 1.0,
            methods_labels_colors_dashes=[
                (imagenet_results[arch].cohen_training_N10000, 'White-box', 'b', [1, 0]),] + surrogate_models + [
                (imagenet_results[arch].all_imagenet_denoising_obj_denoisers, 'MSE', 'r', [2, 4]),
                (imagenet_results[arch].imagenet_no_denoiser_N10000, 'No denoiser', 'k', [5, 1]),
            ],
            sigma=1.00)

    ##################################################################################################         
    # VISION API
    google_api_mse = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_mse/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_no_noise = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/no_denoiser/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_mse = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_mse/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_mse_1k = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_mse/{0:.2f}_1k/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_no_noise = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/no_denoiser/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_mse = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_mse/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_no_noise = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/no_denoiser/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_mse = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_mse/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_no_noise = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/no_denoiser/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_clf_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_clf_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_clf_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_clf_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_clf_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_clf_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_clf_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_clf_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_clf_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_clf_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_clf_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_clf_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_classification_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_smooth_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_smooth_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    azure_api_smooth_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/azure/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_smooth_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_smooth_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    aws_api_smooth_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/aws/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_smooth_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_smooth_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    clarifai_api_smooth_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/clarifai/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_smooth_resnet18 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet18/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_smooth_resnet34 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet34/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    google_api_smooth_resnet50 = [
            Line(ApproximateAccuracy_API("data/certify/vision_api/google/imagenet_denoiser_smoothness_obj_adam_1e-5_20epochs/resnet50/{0:.2f}/log.txt".format(noise)), "$\sigma = {0:.2f}$".format(noise))
        for noise in [0.12, 0.25]
        ]

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/denoiser_finetune_smooth_res18_vs_denoiser_mse", '', 0.6, 
        methods=azure_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=azure_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/denoiser_finetune_smooth_res34_vs_denoiser_mse", '', 0.6, 
        methods=azure_api_smooth_resnet34, label='Stab+MSE on Resnet34',
        methods_base=azure_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/denoiser_finetune_smooth_res50_vs_denoiser_mse", '', 0.6, 
        methods=azure_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=azure_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/denoiser_finetune_smooth_res18_vs_denoiser_mse", '', 0.6, 
        methods=google_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=google_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/denoiser_finetune_smooth_res34_vs_denoiser_mse", '', 0.6, 
        methods=google_api_smooth_resnet34, label='Stab+MSE on Resnet34',
        methods_base=google_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/denoiser_finetune_smooth_res50_vs_denoiser_mse", '', 0.6, 
        methods=google_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=google_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/denoiser_finetune_smooth_res18_vs_denoiser_mse", '', 0.6, 
        methods=aws_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=aws_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/denoiser_finetune_smooth_res34_vs_denoiser_mse", '', 0.6, 
        methods=aws_api_smooth_resnet34, label='Stab+MSE on Resnet34',
        methods_base=aws_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/denoiser_finetune_smooth_res50_vs_denoiser_mse", '', 0.6, 
        methods=aws_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=aws_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/denoiser_finetune_smooth_res18_vs_denoiser_mse", '', 0.6, 
        methods=clarifai_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=clarifai_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/denoiser_finetune_smooth_res34_vs_denoiser_mse", '', 0.6, 
        methods=clarifai_api_smooth_resnet34, label='Stab+MSE on Resnet34',
        methods_base=clarifai_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/denoiser_finetune_smooth_res50_vs_denoiser_mse", '', 0.6, 
        methods=clarifai_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=clarifai_api_mse, label_base="MSE")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/smooth_vs_clf_resnet18", '', 0.6, 
        methods=azure_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=azure_api_clf_resnet18, label_base="Clf+MSE on ResNet18")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/smooth_vs_clf_resnet34", '', 0.6, 
        methods=azure_api_smooth_resnet34, label='Stab+MSE on ResNet34',
        methods_base=azure_api_clf_resnet34, label_base="Clf+MSE on ResNet34")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/smooth_vs_clf_resnet50", '', 0.6, 
        methods=azure_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=azure_api_clf_resnet50, label_base="Clf+MSE on ResNet50")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/smooth_vs_clf_resnet18", '', 0.6, 
        methods=google_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=google_api_clf_resnet18, label_base="Clf+MSE on ResNet18")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/smooth_vs_clf_resnet34", '', 0.6, 
        methods=google_api_smooth_resnet34, label='Stab+MSE on ResNet34',
        methods_base=google_api_clf_resnet34, label_base="Clf+MSE on ResNet34")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/google/smooth_vs_clf_resnet50", '', 0.6, 
        methods=google_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=google_api_clf_resnet50, label_base="Clf+MSE on ResNet50")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/smooth_vs_clf_resnet18", '', 0.6, 
        methods=aws_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=aws_api_clf_resnet18, label_base="Clf+MSE on ResNet18")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/smooth_vs_clf_resnet34", '', 0.6, 
        methods=aws_api_smooth_resnet34, label='Stab+MSE on ResNet34',
        methods_base=aws_api_clf_resnet34, label_base="Clf+MSE on ResNet34")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/aws/smooth_vs_clf_resnet50", '', 0.6, 
        methods=aws_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=aws_api_clf_resnet50, label_base="Clf+MSE on ResNet50")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/smooth_vs_clf_resnet18", '', 0.6, 
        methods=clarifai_api_smooth_resnet18, label='Stab+MSE on ResNet18',
        methods_base=clarifai_api_clf_resnet18, label_base="Clf+MSE on ResNet18")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/smooth_vs_clf_resnet34", '', 0.6, 
        methods=clarifai_api_smooth_resnet34, label='Stab+MSE on ResNet34',
        methods_base=clarifai_api_clf_resnet34, label_base="Clf+MSE on ResNet34")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/clarifai/smooth_vs_clf_resnet50", '', 0.6, 
        methods=clarifai_api_smooth_resnet50, label='Stab+MSE on ResNet50',
        methods_base=clarifai_api_clf_resnet50, label_base="Clf+MSE on ResNet50")

    azure_api_smooth_total = azure_api_smooth_resnet18 + azure_api_smooth_resnet34 +  azure_api_smooth_resnet50
    clarifai_api_smooth_total = clarifai_api_smooth_resnet18 + clarifai_api_smooth_resnet34 +  clarifai_api_smooth_resnet50
    google_api_smooth_total = google_api_smooth_resnet18 + google_api_smooth_resnet34 +  google_api_smooth_resnet50
    aws_api_smooth_total = aws_api_smooth_resnet18 + aws_api_smooth_resnet34 +  aws_api_smooth_resnet50

    plot_certified_accuracy_per_sigma_against_baseline_finetune(
        "analysis/plots/vision_api/azure/total_comparison", '', 0.6, 
        methods=azure_api_mse, label="MSE",
        methods_finetune=azure_api_smooth_total, label_finetune="Stab+MSE best",
        methods_base=azure_api_no_noise, label_base="No Denoiser")

    plot_certified_accuracy_per_sigma_against_baseline_finetune(
        "analysis/plots/vision_api/google/total_comparison", '', 0.6, 
        methods=google_api_mse, label="MSE",
        methods_finetune=google_api_smooth_total, label_finetune="Stab+MSE best",
        methods_base=google_api_no_noise, label_base="No Denoiser")

    plot_certified_accuracy_per_sigma_against_baseline_finetune(
        "analysis/plots/vision_api/aws/total_comparison", '', 0.6, 
        methods=aws_api_mse, label="MSE",
        methods_finetune=aws_api_smooth_total, label_finetune="Stab+MSE best",
        methods_base=aws_api_no_noise, label_base="No Denoiser")

    plot_certified_accuracy_per_sigma_against_baseline_finetune(
        "analysis/plots/vision_api/clarifai/total_comparison", '', 0.6, 
        methods=clarifai_api_mse, label="MSE",
        methods_finetune=clarifai_api_smooth_total, label_finetune="Stab+MSE best",
        methods_base=clarifai_api_no_noise, label_base="No Denoiser")

    plot_certified_accuracy_per_sigma_against_baseline(
        "analysis/plots/vision_api/azure/1k_vs_100", '', 0.6, 
        methods=azure_api_mse_1k, label='MSE with 1k',
        methods_base=azure_api_mse, label_base="MSE with 100")

########################################################################################
# Latex
    for arch in imagenet_archs:
        latex_table_certified_accuracy_upper_envelope(
            "analysis/latex/fullAccess_imagenet_certified_outer_envelop_{}_denoisers".format(arch), 0.25, 1.5, 0.25, 
                imagenet_results[arch].all_imagenet_stability_obj_N10000[arch]
                )
        latex_table_certified_accuracy_upper_envelope(
            "analysis/latex/queryAccess_imagenet_certified_outer_envelop_{}_denoisers".format(arch), 0.25, 1.5, 0.25, 
                sum([imagenet_results[arch].all_imagenet_stability_obj_N10000[b] for b in set(imagenet_archs) - set([arch])], [])
                )
        latex_table_certified_accuracy_upper_envelope(
            "analysis/latex/imagenet_certified_outer_envelop_{}_no_denoisers".format(arch), 0.25, 1.5, 0.25, 
                imagenet_results[arch].imagenet_no_denoiser_N10000)   

        latex_table_certified_accuracy_upper_envelope(
            "analysis/latex/imagenet_certified_outer_envelop_{}_whitebox".format(arch), 0.25, 1.5, 0.25, 
                imagenet_results[arch].cohen_training_N10000)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/cifar10_certified_outer_envelop_no_denoisers", 0.25, 1.5, 0.25, 
            cifar_no_denoiser_N10000)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/cifar10_certified_outer_envelop_whitebox", 0.25, 1.5, 0.25, 
            all_cifar_cohen_N10000)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/queryAccess_cifar10_certified_outer_envelop", 0.25, 1.5, 0.25, 
            all_exp_resnet110_queryAccess_cifar10_stability)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/fullAccess_cifar10_certified_outer_envelop", 0.25, 1.5, 0.25, 
            all_exp_resnet110_fullAccess_cifar10_stability)   
