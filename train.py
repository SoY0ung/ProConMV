import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import random
import numpy as np
import torch
import pandas as pd
import copy
import torchvision.transforms as transform
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, hamming_loss
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Multiview_Concept_Mask_ImgDataset
from models.FLoss import FocalLoss
from models.model import MVCBM3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # Set to cpu for debugging


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(model_name, N_EPOCHS=100, LR=0.0001, depth=12, head=9, num_classes=5):
    # DEBUG ON
    print(f'[INFO] Instance name: {model_name} <LR,EPOCHS,DEPTH,HEAD>')
    model = MVCBM3(in_chans=4, concept_classes=6, num_classes=5, last_channel=512,
                    dim=128, device=device).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    criterion = FocalLoss(gamma=GAMMA)
    criterion_multi_label = nn.BCEWithLogitsLoss()
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    inter_val = 1

    model.to(device)
    best_acc = 0
    for epoch in range(N_EPOCHS):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        train_bar = tqdm(train_loader)

        for i, (img, concept_label, label) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            concept_label = concept_label.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            B, V, C, H, W = img.size()

            concept_pred, out = model(img)

            concept_label = concept_label.reshape(-1, 6)
            concept_pred = concept_pred.reshape(-1, 6)
            loss = criterion(out, label) + ALPHA * criterion_multi_label(concept_pred, concept_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_acc += (out.argmax(dim=1) == label).sum()
            train_epoch_loss += loss.item()
            scheduler.step(epoch + i / len(train_loader))
            optimizer.step()

        train_loss_mean = train_epoch_loss / len(train_loader)
        train_acc_mean = train_epoch_acc / (len(train_dataset) * NUM_VIEW)

        train_loss.append(train_loss_mean)
        train_acc.append(train_acc_mean.cpu())

        print('{} train loss: {:.3f} train acc: {:.3f}  lr:{}'.format(epoch, train_loss_mean,
                                                                      train_acc_mean,
                                                                      optimizer.param_groups[-1]['lr']))
        if (epoch + 1) % inter_val == 0:
            val_acc_mean, val_loss_mean = val_model(model, test_loader, criterion, device)

            if val_acc_mean > best_acc:
                model.cpu()
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                best_acc = val_acc_mean
                save_path = f'./weights/proconmv.pth'
                torch.save(model.state_dict(), save_path)
                print(f"model saved at weights {save_path}")
            valid_loss.append(val_loss_mean)
            valid_acc.append(val_acc_mean.cpu())
        print('now_best:{:.6f}'.format(best_acc))

    print("best val acc:", best_acc)



def val_model(model, valid_loader, criterion, device):
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    model.eval()
    valid_bar = tqdm(valid_loader)
    gamma = 1
    for i, (img, concept_label, label) in enumerate(valid_bar):
        img = img.to(device)
        label = label.to(device)
        concept_label = concept_label.to(device)
        num_classes = 5
        B, V, C, H, W = img.size()
        # img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        with torch.no_grad():
            concept_out, out = model(img)

        loss = criterion(out, label)
        valid_epoch_loss += loss.item()
        valid_epoch_acc += (out.argmax(dim=1) == label).sum()

    val_acc_mean = valid_epoch_acc / (len(valid_dataset) * NUM_VIEW)
    val_loss_mean = valid_epoch_loss / len(valid_loader)

    print('valid loss: {:.3f} valid acc: {:.3f}'.format(val_loss_mean, val_acc_mean))

    return val_acc_mean, val_loss_mean


from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, hamming_loss, cohen_kappa_score, accuracy_score, precision_score, recall_score


def val_model2(model, valid_loader, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []
    all_concept_preds = []
    all_concept_labels = []

    criterion_concept = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for img, concept_label, label in tqdm(valid_loader, desc='Validating'):
            img = img.to(device)
            label = label.to(device)
            concept_label = concept_label.to(device)
            B, V, C, H, W = img.size()

            concept_out, out = model(img)

            loss_cls = criterion_cls(out, label)

            concept_out = concept_out.reshape(B, -1)
            concept_label = concept_label.reshape(B, -1)
            loss_concept = criterion_concept(concept_out, concept_label)

            loss = loss_cls + 0.5 * loss_concept
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            total_correct += (preds == label).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_concept_preds.append(torch.sigmoid(concept_out).cpu().numpy())
            all_concept_labels.append(concept_label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    confmat = confusion_matrix(all_labels, all_preds)

    concept_pred_arr = np.concatenate(all_concept_preds, axis=0)
    concept_true_arr = np.concatenate(all_concept_labels, axis=0)

    map_score = average_precision_score(concept_true_arr, concept_pred_arr, average='macro')
    hamming = hamming_loss(concept_true_arr, concept_pred_arr > 0.5)
    f1_concept_macro = f1_score(concept_true_arr, concept_pred_arr > 0.5, average='macro')

    print(f'[Validation] Loss: {total_loss / len(valid_loader):.4f}')
    print(f'  Acc: {acc:.4f} | F1_macro: {f1_macro:.4f} | F1_micro: {f1_micro:.4f}')
    print(f'  Precision: {precision:.4f} | Recall: {recall:.4f} | Kappa: {kappa:.4f}')
    # print(f'  AUC: {auc_score:.4f} | AUPR: {aupr_score:.4f}')
    print(f'  Concept mAP: {map_score:.4f} | Hamming: {hamming:.4f} | Concept F1_macro: {f1_concept_macro:.4f}')
    print('  Confusion Matrix:\n', confmat)

    return acc, total_loss / len(valid_loader)


if __name__ == '__main__':
    seed_everything(1001)
    # general global variables
    DATA_PATH = "./data/mfiddr"
    TRAIN_PATH = "./data/mfiddr/train_process"
    MASK_PATH = "./data/mfiddr/train_mask_cleaned"
    TEST_PATH = "./data/mfiddr/test_process"
    MASK_PATH2 = "./data/mfiddr/test_mask_cleaned"
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = 'weights'
    NUM_VIEW = 1
    IMAGE_SIZE = 256
    LR = 0.00001
    N_EPOCHS = 100
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 16

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default='0.2', help='loss function weight')
    parser.add_argument('--gamma', type=float, default='2', help='focal loss gamma')
    args = parser.parse_args()
    ALPHA = args.alpha
    GAMMA = args.gamma
    print(f'[INFO] ALPHA: {ALPHA}, GAMMA: {GAMMA}')

    train_csv_path = './train.csv'
    # assert os.path.exists(train_csv_path), '{} path is not exists...'.format(train_csv_path)
    test_csv_path = './test.csv'
    test_df = pd.read_csv(test_csv_path)

    all_data = pd.read_csv(train_csv_path)
    all_data.head()

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.3),
        transform.RandomVerticalFlip(p=0.3),
        # transform.RandomResizedCrop(IMAGE_SIZE),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_valid = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = Multiview_Concept_Mask_ImgDataset(TRAIN_PATH, MASK_PATH, all_data, transform=transform_train)
    valid_dataset = Multiview_Concept_Mask_ImgDataset(TEST_PATH, MASK_PATH2, test_df, transform=transform_test)
    test_dataset = Multiview_Concept_Mask_ImgDataset(TEST_PATH, MASK_PATH2, test_df, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)

    print(f'[INFO] Batch Size: {BATCH_SIZE}')
    print(f'[INFO] Training set:{len(train_dataset)}, Validation set:{len(valid_dataset)}, Testing set:{len(test_dataset)}')
    print(f'[INFO] CUDA device:{device}')
    main(model_name=f'main_{LR}_{N_EPOCHS}_d{DEPTH}_h{HEAD}', N_EPOCHS=N_EPOCHS, LR=LR, depth=DEPTH, head=HEAD)
