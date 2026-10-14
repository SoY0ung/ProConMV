import numpy as np
import torch
import pandas as pd
import torchvision.transforms as transform
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, roc_auc_score, \
    label_ranking_average_precision_score, multilabel_confusion_matrix, average_precision_score, label_ranking_loss, \
    coverage_error
from sklearn.preprocessing import label_binarize
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from dataset import Multiview_Concept_Mask_ImgDataset
from models.model import MVCBM3


def test_model2(netname, model, test_loader, dataset_size, criterion, device,compute_multi_label_metrics=True):
    n_class = 5
    print('dataset_size', dataset_size)
    since = time.time()
    # roc matrix
    roc_matrix = pd.DataFrame(columns=['label', '0', '1', '2', '3', '4'])

    if compute_multi_label_metrics:
        all_multi_preds = []
        all_multi_targets = []
        all_outputs_prob = []
    # fusion matrix
    total_outs = torch.tensor([])
    total_labels = torch.tensor([])
    FM = np.zeros((n_class, n_class))
    tp = [0] * n_class
    tn = [0] * n_class
    fp = [0] * n_class
    fn = [0] * n_class
    precision = [0] * n_class
    recall = [0] * n_class
    specificity = [0] * n_class
    f1 = [0] * n_class
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    running_loss = 0.0
    running_corrects = 0
    model.eval()
    valid_bar = tqdm(test_loader)
    for i, (img, concept_label, label) in enumerate(valid_bar):
        img = img.to(device)
        concept_label = concept_label.to(device)
        # labels = label.to(device)
        B, V, C, H, W = img.size()
        # img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        with torch.no_grad():
            concept_pred, cls_pred = model(img)
        outputs = cls_pred.cpu()
        loss = criterion(outputs, label)

        total_labels = torch.cat((total_labels, label), dim=0) if total_labels.size(0) > 0 else label
        total_outs = torch.cat((total_outs, outputs.detach()), dim=0) if total_outs.size(0) > 0 else outputs.detach()


        running_loss += loss.item()

        if compute_multi_label_metrics:
            concept_label = concept_label.reshape(-1, 6)
            concept_pred = concept_pred.reshape(-1, 6)
            outputs_prob = torch.sigmoid(concept_pred)
            multi_preds = (outputs_prob >= 0.5).float()
            all_multi_preds.append(multi_preds.cpu().numpy())
            # concept_label = torch.round(concept_label.mean(dim=1))
            all_multi_targets.append(concept_label.cpu().numpy())
            all_outputs_prob.append(outputs_prob.cpu().numpy())
    preds = torch.argmax(total_outs, dim=1)
    outputs_softmax = torch.softmax(total_outs, 1)
    labels = total_labels
    # print(classification_report(labels,preds))
    cls_statis = [torch.sum(labels == 0), torch.sum(labels == 1), sum(labels == 2), sum(labels == 3), sum(labels == 4)]

    running_corrects += torch.sum(preds == labels)
    # running_corrects+=(output1.argmax(dim=1) == label).sum()

    for batch_i in range(len(labels)):

        # save roc matrix
        roc_df = {'label': labels[batch_i].numpy(),
                  '0': outputs_softmax[batch_i][0].numpy(),
                  '1': outputs_softmax[batch_i][1].numpy(),
                  '2': outputs_softmax[batch_i][2].numpy(),
                  '3': outputs_softmax[batch_i][3].numpy(),
                  '4': outputs_softmax[batch_i][4].numpy()
                  }
        roc_matrix = pd.concat([roc_matrix, pd.DataFrame([roc_df])], ignore_index=True)

        # fusion matrix
        predict_label = preds[batch_i]
        true_label = labels[batch_i]
        # FM[predict_label][true_label] = FM[predict_label][true_label] + 1
        FM[true_label][predict_label] = FM[true_label][predict_label] + 1

        for label in range(n_class):
            p_or_n_from_pred = (label == preds[batch_i])
            p_or_n_from_label = (label == labels[batch_i])

            if p_or_n_from_pred == 1 and p_or_n_from_label == 1:
                tp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 0:
                tn[label] += 1
            if p_or_n_from_pred == 1 and p_or_n_from_label == 0:
                fp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 1:
                fn[label] += 1

    # each class test results
    for label in range(n_class):
        precision[label] = tp[label] / (tp[label] + fp[label] + 1e-8)
        recall[label] = tp[label] / (tp[label] + fn[label] + 1e-8)
        specificity[label] = tn[label] / (tn[label] + fp[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        print('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle = open(netname + '_test_result.txt', 'a')
        fileHandle.write('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}\n'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle.close()

        # save Fusion Matric
    print('\nFusion Matrix:')
    print(FM)
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write('\nFusion Matrix:\n')
    for f_i in FM:
        fileHandle.write(str(f_i) + '\r\n')
    fileHandle.close()

    # save roc data
    roc_matrix.to_csv(netname + '_roc_data.csv', encoding='gbk')

    # calculate the Kappa
    pe0 = (tp[0] + fn[0]) * (tp[0] + fp[0])
    pe1 = (tp[1] + fn[1]) * (tp[1] + fp[1])
    pe2 = (tp[2] + fn[2]) * (tp[2] + fp[2])
    pe3 = (tp[3] + fn[3]) * (tp[3] + fp[3])
    pe4 = (tp[4] + fn[4]) * (tp[4] + fp[4])
    pe = (pe0 + pe1 + pe2 + pe3 + pe4) / (dataset_size * dataset_size)
    pa = (tp[0] + tp[1] + tp[2] + tp[3] + tp[4]) / dataset_size
    kappa = (pa - pe) / (1 - pe)

    # overall test results
    test_epoch_loss = running_loss / dataset_size
    test_epoch_acc = running_corrects / dataset_size
    overall_precision = sum([cls_statis[i] * p for i, p in enumerate(precision)]) / sum(cls_statis)
    overall_recall = sum([cls_statis[i] * r for i, r in enumerate(recall)]) / sum(cls_statis)

    overall_specificity = sum([cls_statis[i] * s for i, s in enumerate(specificity)]) / sum(cls_statis)
    overall_f1 = sum([cls_statis[i] * f for i, f in enumerate(f1)]) / sum(cls_statis)

    # macro AUC
    y_true = roc_matrix['label'].astype(int).values
    y_score = roc_matrix[['0', '1', '2', '3', '4']].astype(float).values
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    overall_auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')

    elapsed_time = time.time() - since
    print(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, avg_precision: {:.4f},avg_recall: {:.4f},avg_specificity: {:.4f},avg_f1: {:.4f}, avg_auc: {:.4f}, Total elapsed time: {:.4f} '.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1, overall_auc,
            elapsed_time))
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, all_precision: {:.4f},all_recall: {:.4f},all_specificity: {:.4f},all_f1: {:.4f}, all_auc: {:.4f}, Total elapsed time: {:.4f} \n'.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1, overall_auc,
            elapsed_time))
    fileHandle.close()


    if compute_multi_label_metrics:
        print("\n====== Multi-label Metrics (concept_classes=6) ======")

        all_multi_preds = np.vstack(all_multi_preds)
        all_multi_targets = np.vstack(all_multi_targets)
        all_outputs_prob = np.vstack(all_outputs_prob)
        # print(all_multi_preds.size,all_multi_targets.size)

        multi_label_conf_matrix = multilabel_confusion_matrix(all_multi_targets, all_multi_preds)
        print("Multilabel Confusion Matrices (per class):")
        print(multi_label_conf_matrix)

        # -------------------------------------------------------------------
        # 1) Precision / Recall / F1 (micro / macro / weighted)
        # -------------------------------------------------------------------
        # micro_precision = precision_score(all_multi_targets, all_multi_preds, average='micro', zero_division=0)
        # micro_recall = recall_score(all_multi_targets, all_multi_preds, average='micro', zero_division=0)
        micro_f1 = f1_score(all_multi_targets, all_multi_preds, average='micro', zero_division=0)

        # macro_precision_ml = precision_score(all_multi_targets, all_multi_preds, average='macro', zero_division=0)
        # macro_recall_ml = recall_score(all_multi_targets, all_multi_preds, average='macro', zero_division=0)
        macro_f1_ml = f1_score(all_multi_targets, all_multi_preds, average='macro', zero_division=0)

        # weighted_precision_ml = precision_score(all_multi_targets, all_multi_preds, average='weighted',
        # zero_division=0) weighted_recall_ml = recall_score(all_multi_targets, all_multi_preds, average='weighted',
        # zero_division=0)
        weighted_f1_ml = f1_score(all_multi_targets, all_multi_preds, average='weighted', zero_division=0)

        # print(f"Multi-label Micro Precision:  {micro_precision:.4f}")
        # print(f"Multi-label Micro Recall:     {micro_recall:.4f}")
        print(f"Multi-label Micro F1:         {micro_f1:.4f}")

        # print(f"Multi-label Macro Precision:  {macro_precision_ml:.4f}")
        # print(f"Multi-label Macro Recall:     {macro_recall_ml:.4f}")
        print(f"Multi-label Macro F1:         {macro_f1_ml:.4f}")

        # print(f"Multi-label Weighted Precision:  {weighted_precision_ml:.4f}")
        # print(f"Multi-label Weighted Recall:     {weighted_recall_ml:.4f}")
        print(f"Multi-label Weighted F1:         {weighted_f1_ml:.4f}")

        # -------------------------------------------------------------------
        # 2) Ranking Loss (RL)
        # -------------------------------------------------------------------


        try:
            rl = label_ranking_loss(all_multi_targets, all_outputs_prob)
        except ValueError:
            rl = float('nan')
        print(f"Ranking Loss (RL): {rl:.4f}")

        # -------------------------------------------------------------------
        # 3) Hamming Loss (HL)
        # -------------------------------------------------------------------
        hl = hamming_loss(all_multi_targets, all_multi_preds)
        print(f"Hamming Loss: {hl:.4f}")
        ce = coverage_error(all_multi_targets, all_multi_preds)
        print(f"Cov Loss: {ce:.4f}")
        # -------------------------------------------------------------------
        overall_accuracy = (all_multi_preds == all_multi_targets).mean()
        print(f"Overall Accuracy (label-wise mean): {overall_accuracy:.4f}")
        # 4) Subset Accuracy
        # -------------------------------------------------------------------


        subset_acc = accuracy_score(all_multi_targets, all_multi_preds)
        print(f"Subset Accuracy: {subset_acc:.4f}")

        # -------------------------------------------------------------------
        # 5) AUC (macro) & AUPR (macro)
        # -------------------------------------------------------------------

        try:
            auc_macro = roc_auc_score(all_multi_targets, all_outputs_prob, average='macro')
        except ValueError:
            auc_macro = float('nan')
        print(f"Macro AUC: {auc_macro:.4f}")

        try:
            aupr_macro = average_precision_score(all_multi_targets, all_outputs_prob, average='macro')
        except ValueError:
            aupr_macro = float('nan')
        print(f"Macro AUPR: {aupr_macro:.4f}")

        # -------------------------------------------------------------------
        # 6) Average Precision (AP) & (mAP)
        # -------------------------------------------------------------------

        ap_list = []
        n_labels = all_multi_targets.shape[1]
        for i in range(n_labels):
            y_true_i = all_multi_targets[:, i]
            y_prob_i = all_outputs_prob[:, i]


            try:
                ap_i = average_precision_score(y_true_i, y_prob_i)
                ap_list.append(ap_i)
                print(f"Classes {i}: {ap_i:.4f}")
            except ValueError:
                ap_list.append(np.nan)
        valid_aps = [ap for ap in ap_list if not np.isnan(ap)]
        if len(valid_aps) > 0:
            map_score = np.mean(valid_aps)
        else:
            map_score = float('nan')
        print(f"Mean Average Precision (mAP) across all classes: {map_score:.4f}")

        # -------------------------------------------------------------------
        # 7) Label Ranking Average Precision Score (LR-APS)
        # -------------------------------------------------------------------


        try:
            lr_map = label_ranking_average_precision_score(all_multi_targets, all_outputs_prob)
        except ValueError:
            lr_map = float('nan')
        print(f"Label Ranking Average Precision Score (LR-APS): {lr_map:.4f}")
    return (test_epoch_loss, test_epoch_acc)

if __name__ == '__main__':
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    DATA_PATH = "./data/mfiddr"
    TEST_PATH = "./data/mfiddr/test_process"
    MASK_PATH2 = "./data/mfiddr/train_mask_cleaned"
    BATCH_SIZE = 16
    IMAGE_SIZE = 256
    MODELPATH = "./weights/proconmv.pth"

    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    checkpoint = torch.load(MODELPATH, map_location=device)
    model = MVCBM3(in_chans=4, concept_classes=6, num_classes=5, last_channel=512, dim=128, device=device).to(device)
    model.load_state_dict(checkpoint)
    test_csv_path = './test.csv'
    test_df = pd.read_csv(test_csv_path)
    test_dataset = Multiview_Concept_Mask_ImgDataset(TEST_PATH, MASK_PATH2, test_df, transform=transform_test)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24, pin_memory=False)

    # device = torch.device('cpu')
    model.to(device)

    test_model2('final', model, test_loader, len(test_dataset),
                nn.CrossEntropyLoss(reduction='sum'), device)
