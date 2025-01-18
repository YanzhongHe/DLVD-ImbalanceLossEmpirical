# On the Value of Imbalance Loss Functions in Enhancing Deep Learning-Based Vulnerability Detection

## Introduction

Repository for the paper: On the Value of Imbalance Loss Functions in Enhancing Deep Learning-Based Vulnerability Detection.

**Abstract:** Software vulnerability detection is crucial in software engineering and information security, and deep learning has been demonstrated to be effective in this domain. However, the class imbalance issue, where non-vulnerable code snippets vastly outnumber vulnerable ones, hinders the performance of Deep Learning-based Vulnerability Detection (DLVD) models. Although some recent research has explored the use of imbalance loss functions to address this issue and enhance model efficacy, they have primarily focused on a limited selection of imbalance loss functions, leaving many others unexplored. Therefore, their conclusions about the most effective imbalance loss function may be biased and inconclusive. To fill this gap, we first conduct a comprehensive literature review of 119 DLVD studies, focusing on the loss functions used by these models. We then assess the effectiveness of nine imbalance loss functions alongside Cross-Entropy Loss (the standard balanced loss function) on two DLVD models across four public vulnerability datasets. Our evaluation incorporates four performance metrics, with results analyzed using the Scott-Knott Effect Size Difference test. Furthermore, we employ interpretable analysis to elucidate the impact of loss functions on model performance. Our findings provide key insights for DLVD, which mainly include the following: the LineVul model consistently outperforms the ReVeal model; Label Distribution Aware Margin Loss achieves the highest Precision, while Logit Adjustment Loss (LALoss) yields the best Recall; Class Balanced Focal Loss (CBFocalLoss) excels in comprehensive performance on extremely imbalanced datasets; and LALoss is optimal for nearly balanced datasets. We recommend using LineVul with either CBFocalLoss or LALoss to enhance DLVD outcomes.

## Environment Setup

```

$ pip intsall numpy

$ pip install torch

$ pip install transformers

$ pip install sklearn

$ pip install scikit-learn

$ pip install balanced-loss

$ pip install pandas

$ pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html

$ pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

$ pip install torch_geometric

$ pip install tqdm

$ pip install imbalanced-learn

$ pip install imbalanced-ensemble

```

## DLVD_LineVul

### Text Dataset

We utilize the cleaned versions of **Devign**, **Big-Vul**, and **Juliet** provided by Roland et al. [Here](https://figshare.com/articles/software/Reproduction_Package_for_Data_Quality_for_Software_Vulnerability_Datasets_/20499924) is the cleaned version data link. We also used the original **ReVeal** dataset as it does not have a clean version. To adapt to the **LineVul** basic detection models, we processed the four datasets into **text datasets**.

Please click [this link](https://figshare.com/s/5483f706b04cb9a66fa3) to download the `DLVD_LineVul/data.zip` data and place it in the `DLVD_LineVul/` path. Use the `unzip data.zip` command to get the VD_data folder.

You can get the original data set by running the code below:

```

$ cd DLVD_LineVul

$ unzip data.zip

```

### CodeBert

[Here](https://huggingface.co/microsoft/codebert-base) is the official CodeBert. To facilitate reproduction, we recommend that you download the `pytorch_model.bin` file and place it in the `DLVD_LineVul/bert/` directory.

## DLVD_ReVeal

### Graph Dataset

We utilize the cleaned versions of **Devign**, **Big-Vul**, and **Juliet** provided by Roland et al. [Here](https://figshare.com/articles/software/Reproduction_Package_for_Data_Quality_for_Software_Vulnerability_Datasets_/20499924) is the cleaned version data link. We also used the original **ReVeal** dataset as it does not have a clean version. To adapt to the **ReVeal** basic detection models, we processed the four datasets into **graph datasets**.

For graph datasets,  Many works utilize Joern to extract graphs from source code. However, due to the difficulty in configuring the Joern environment, we provide the preprocessed graph dataset using Joern in `DLVD_ReVeal/ReVeal/VD_data/`.

Please click [this link](https://figshare.com/s/5483f706b04cb9a66fa3) to download the `DLVD_ReVeal/ReVeal/VD_data.zip` data and place it in the `DLVD_ReVeal/ReVeal/` path. Use the `unzip VD_data.zip` command to get the VD_data folder.

You can use the following method to get the original dataset. For example, the reveal dataset is demonstrated as follows:

```

$ cd DLVD_ReVeal/ReVeal/VD_data/reveal

$ unzip reveal-1.0.zip

$ python spilt.py


```

## Imbalance Loss Functions

The imbalance loss functions code scripts are placed in `DLVD_LineVul/unbalanced_loss` and `DLVD_ReVeal/ReVeal/unbalanced_loss`.

## Run DLVD

### Run DLVD_LineVul

Run the following code script.

```
$ cd DLVD_LineVul

$ python main_linevul.py
    --loss_comment {LOSS_COMMENT}
    --comment {COMMENT}

```

`--loss_comment`: Choose the loss functions on which to train. `celoss`, `LAloss`,`LDAMloss`,`WCEloss`,`GHMloss`,`focalloss`, `CBceloss`, `CBfocalloss`,`DiceLoss`, `DSCLoss`.

`--comment`: The name of the best training model.

### Run DLVD_ReVeal

Run the following code script.

```
$ cd DLVD_ReVeal

$ python build.py
    --dataset {DATASET_NAME}
    --label_rate 1.0
    --stopper_mode {STOPPER_MODE}
    --loss_fct {LOSS_FCT}
    --comment {COMMENT}

```

`--dataset`: Choose the dataset on which to train. `Devign`, `reveal`,`big-vul`, or `Juliet`.

`--stopper_mode`: Stop training when there is no improvement in the validation set's F1/accuracy. `f1` or `acc`. We use the `f1` as the default.

`--loss_fct`: Choose the loss functions on which to train. `celoss`, `LAloss`,`LDAMloss`,`WCEloss`,`GHMloss`,`focalloss`, `CBceloss`, `CBfocalloss`,`DiceLoss`, `DSCLoss`.

`--comment`: The name of the best training model.

After training is completed, the best model will be stored in the path `{BASE_MODEL}/best_model/{DATASET_NAME}/1.0/{STOPPER_MODE}/`.

### Run LIT

Run the following code script to perform model interpretability analysis. It is worth noting that the corresponding DLVD model needs to be obtained first.

```
$ cd LIT/code

$ python all_lit.py

```
