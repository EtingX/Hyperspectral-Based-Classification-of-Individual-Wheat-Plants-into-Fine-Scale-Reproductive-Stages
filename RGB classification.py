import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, efficientnet_v2_s
from torchvision.models.swin_transformer import Swin_V2_T_Weights
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
import time
import wandb
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report, f1_score, confusion_matrix
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
import shutil
import torch.optim as optim
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
from timm.data.mixup import Mixup
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os


from torch.utils.data import Dataset

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FlexibleClassifier(nn.Module):
    def __init__(self, model_type, num_classes):
        super(FlexibleClassifier, self).__init__()
        self.model_type = model_type.lower()

        if self.model_type == 'swinv2':
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
            self.base_model = swin_v2_t(weights=weights)
            feature_size = self.base_model.head.in_features  # Dynamically get the feature size
            self.base_model.head = nn.Identity()  # Remove the classifier head

        elif self.model_type == 'efficientnet':
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.base_model = efficientnet_v2_s(weights=weights)
            feature_size = self.base_model.classifier[-1].in_features

            self.base_model.classifier = nn.Identity()

        else:
            raise ValueError("Unsupported model type. Choose 'swinv2' or 'efficientnet'.")

        # Define the new classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print_val_loss_min = self.val_loss_min * -1
            print_val_loss = val_loss * -1  # -1 because using F1 score, if loss change to 1
            print(f'F1 score increased ({print_val_loss_min:.6f} --> {print_val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')  # save model here

        self.val_loss_min = val_loss


class EMA():
    '''
        strategy can be particularly useful for large models or when aiming
        to improve the stability and generalization of the training process.
    '''

    def __init__(self, model, decay):
        # Initializing the EMA class with a model and a decay rate.
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        # Registering the model's parameters with ema.register() before the training loop.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # Updating the EMA of the parameters after each epoch or update step.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Applying the EMA parameters for evaluation or inference using ema.apply_shadow().
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # Restoring the original parameters with ema.restore() after evaluation to
        # continue training or for other purposes.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch.

    Parameters:
    - model: The feature extraction model.
    - device: The device (CPU or GPU) to train on.
    - train_loader: DataLoader for training data.
    - optimizer: The optimizer.
    - epoch: Current epoch number.
    - use_amp: Flag to use automatic mixed precision.
    - use_ema: Flag to use exponential moving average.
    - criterion_train: Loss function.
    - accuracy: Function to compute accuracy metrics.
    - ema: EMA object if use_ema is True.

    Returns:
    - Average loss and accuracy for the epoch.
    """

    model.train()
    total_targets = []
    total_predictions = []
    # Metrics class
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(f'Total number of samples: {total_num}, Batches: {len(train_loader)}')

    # Progress bar with tqdm
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, data in pbar:
        img, target = data
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img, target = mixup_fn(img, target)
        output = model(img)
        # print(output)
        # print("Output shape:", output.shape)
        # print("Target shape:", target.shape)
        optimizer.zero_grad()
        if use_amp:
            # Automatic mixed precision
            with torch.cuda.amp.autocast():
                loss = criterion_train(output, target)
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            # Apply EMA model parameters for evaluation if enabled
            if use_ema:
                ema.update()
        else:
            loss = criterion_train(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            # Apply EMA model parameters for evaluation if enabled
            if use_ema:
                ema.update()

        # Synchronize GPU operations if necessary
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))

        _, target_indices = torch.max(target, dim=1)
        # Update metrics, only acc1 is useful, as only two classes
        acc1, acc3 = accuracy(output, target_indices, topk=(1, 3))
        acc1_meter.update(acc1.item(), target.size(0))
        acc3_meter.update(acc3.item(), target.size(0))

        # Update progress bar
        pbar.set_postfix({'Loss': loss_meter.avg, 'LR': lr})

        # preds = torch.argmax(output, dim=1).detach().cpu().numpy()
        # labels = target.detach().cpu().numpy()
        # total_predictions.extend(preds)
        # total_targets.extend(labels)

    # f1 = f1_score(total_targets, total_predictions, average='weighted')
    print(f'Epoch: {epoch}\tLoss: {loss_meter.avg:.4f}\tAcc: {acc1_meter.avg:.4f}')
    return loss_meter.avg, acc1_meter.avg


# Val section
@torch.no_grad()
def val(model, device, test_loader):
    """
        Evaluates the model on a test dataset.

        Parameters:
        - model (torch.nn.Module): The main model for feature extraction.

        - device (torch.device): The device to perform the evaluation on (e.g., 'cuda' or 'cpu').
        - test_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.

        Returns:
        - val_list (list): A list of true labels from the test dataset.
        - pred_list (list): A list of predicted labels for the test dataset.
        - loss_meter.avg (float): The average loss computed across all test data.
        - acc (float): The top-1 accuracy percentage across the test dataset.
        - F1 (float): The weighted F1 score based on true and predicted labels.

        This function performs evaluation by iterating through the test dataset, computing predictions and loss, and
        then calculating accuracy metrics including F1 score.

        """
    # global Best_ACC
    global Best_F1 # Tracks the best F1 score across epochs
    model.eval()

    # Metrics class
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()

    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))

    # Stores true and predict labels
    val_list = []
    pred_list = []

    # Apply EMA model parameters for evaluation if enabled
    if use_ema:
        ema.apply_shadow()

    # Iterate over the test dataset
    for data in test_loader:
        img, target = data

        # Record true labels
        for t in target:
            val_list.append(t.data.item())
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Generate predictions
        output = model(img)

        loss = criterion_val(output, target)

        # Determine predicted classes
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())

        # Update accuracy and loss trackers, only acc1 is useful as only two classes
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc3_meter.update(acc5.item(), target.size(0))

    # Restore original model parameters if EMA was applied
    if use_ema:
        ema.restore()
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc3_meter.avg))

    # Compute F1 score
    F1 = f1_score(val_list, pred_list, average='weighted')

    # Check for F1 improvement and save models accordingly
    if F1 > Best_F1 and F1 >= 0.5:
        wandb.save('best_F1_model.h5')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.state_dict(), file_dir + "/" + 'model_F1_' + str(epoch) + '_' + str(round(F1, 3)) +
                       '_' + str(round(loss_meter.avg, 4)) + '.pth')
            torch.save(model.state_dict(), file_dir + '/' + 'F1_best.pth')

        else:
            torch.save(model.state_dict(), file_dir + "/" + 'model_F1_' + str(epoch) + '_' + str(round(F1, 3)) +
                       '_' + str(round(loss_meter.avg, 4)) + '.pth')
            torch.save(model.state_dict(), file_dir + '/' + 'F1_best.pth')
        Best_F1 = F1

    # Save the last model state
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.state_dict(), file_dir + "/" + 'last.pth')
    else:
        torch.save(model.state_dict(), file_dir + "/" + 'last.pth')

    return val_list, pred_list, loss_meter.avg, acc, F1


# Val section
@torch.no_grad()
def test(model, device, test_loader):
    """
        Evaluates the model on a test dataset.

        Parameters:
        - model (torch.nn.Module): The main model for feature extraction.

        - device (torch.device): The device to perform the evaluation on (e.g., 'cuda' or 'cpu').
        - test_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.

        Returns:
        - val_list (list): A list of true labels from the test dataset.
        - pred_list (list): A list of predicted labels for the test dataset.
        - loss_meter.avg (float): The average loss computed across all test data.
        - acc (float): The top-1 accuracy percentage across the test dataset.
        - F1 (float): The weighted F1 score based on true and predicted labels.

        This function performs evaluation by iterating through the test dataset, computing predictions and loss, and
        then calculating accuracy metrics including F1 score.

        """
    # global Best_ACC
    global Best_F1 # Tracks the best F1 score across epochs
    model.eval()

    # Metrics class
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()

    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))

    # Stores true and predict labels
    val_list = []
    pred_list = []

    # Apply EMA model parameters for evaluation if enabled
    if use_ema:
        ema.apply_shadow()

    # Iterate over the test dataset
    for data in test_loader:
        img, target = data

        # Record true labels
        for t in target:
            val_list.append(t.data.item())
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Generate predictions
        output = model(img)

        loss = criterion_val(output, target)

        # Determine predicted classes
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())

        # Update accuracy and loss trackers, only acc1 is useful as only two classes
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc3_meter.update(acc5.item(), target.size(0))

    # Restore original model parameters if EMA was applied
    if use_ema:
        ema.restore()
    acc = acc1_meter.avg
    print('\nTest set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc3_meter.avg))

    # Compute F1 score
    test_F1 = f1_score(val_list, pred_list, average='weighted')

    return val_list, pred_list, loss_meter.avg, acc, test_F1


# 修改sgd_optimizer函数以接受named_parameters列表
def sgd_optimizer(named_params, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


# 修改adam_optimizer函数以接受named_parameters列表
def adam_optimizer(named_params, lr, weight_decay, use_custwd):
    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.Adam(params, lr)
    return optimizer


if __name__ == '__main__':
    '''
    Login to the wandb first and then train the whole process, the RepVgg is RepVGG_B3
    Changing option could be 'optimizer' and 'net architecture'
    All models are in Model_YX.py
    '''

    project = '0757 TOP'
    run_name = '0757 TOP image'
    config = dict(
        learning_rate=1e-3,
        eta_min=1e-4,
        final_learning_rate=1e-4,
        EPOCHS=300,
        architecture="efficientnet", #swinv2 or efficientnet
        optimizer='ADAM',
        opt_momentum=0.9,
        opt_weight_decay=1e-4,
        train_batch_size=32,
        test_batch_size=32,
        DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        use_amp=True,  # 是否使用混合精度
        use_dp=False,  # multiple GPU
        classes=['0', '1', '2'],
        resume=False,
        use_ema=True,
        train_loss='SoftTargetCrossEntropy',
        val_loss='CrossEntropy',
        attention=False,
        early_stop=True,
        schedule_lr='cosine'
    )

    wandb.init(project=project,
               name=run_name,
               config=config,
               mode='online')
    wandb.save("*.pt")
    wandb.watch_called = False

    # 设置全局参数
    model_lr = wandb.config.learning_rate
    train_BATCH_SIZE = wandb.config.train_batch_size
    test_BATCH_SIZE = wandb.config.test_batch_size
    optimizer = wandb.config.optimizer
    EPOCHS = wandb.config.EPOCHS
    DEVICE = wandb.config.DEVICE
    use_amp = wandb.config.use_amp
    use_dp = wandb.config.use_dp
    classes = len(wandb.config.classes)
    resume = wandb.config.resume
    use_ema = wandb.config.use_ema
    early_stop = wandb.config.early_stop
    schedule_lr = wandb.config.schedule_lr
    architecture = wandb.config.architecture

    # save model dir
    localtime = time.asctime(time.localtime(time.time())).split()
    str_time = str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3][0:2])
    # checkpoint folder

    file_dir = 'I:/0757 hyperspectral/RGB/side_model_three' + str(
        wandb.config.architecture) + '/' + str_time

    print(DEVICE)
    model_path = ''

    from torchvision import transforms

    # Transformations for the training set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformations for the validation set
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Path to your dataset
    data_dir = r'I:\0757 hyperspectral\RGB\final_side_dataset\train'


    full_dataset = datasets.ImageFolder(data_dir)
    num_classes = len(full_dataset.classes)
    # Step 3: Split the dataset into training and validation sets
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformDataset(train_dataset, transform=train_transform)
    val_dataset = TransformDataset(val_dataset, transform=val_transform)

    # Step 5: Create data loaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.test_batch_size, shuffle=False)

    # Optional: Print the number of batches in each loader
    print(f'Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}')

    print("Loaded classes:", full_dataset.classes)  # Prints the classes discovered by ImageFolder
    print("Number of images:", len(full_dataset))  # Prints the total number of loaded images
    print("Example data point:", full_dataset[0])  # Prints the first data point

    mixup_fn = Mixup(
        mixup_alpha=0.8,  # Alpha for mixup, typical values are [0.4, 1.0]
        cutmix_alpha=1.0,  # Alpha for cutmix, a variation of mixup
        cutmix_minmax=None,
        prob=1.0,  # Probability to apply mixup or cutmix per batch
        switch_prob=0.5,  # Probability to switch to cutmix instead of mixup
        mode='batch',  # Apply per batch
        label_smoothing=0.1,  # Label smoothing factor
        num_classes=num_classes  # Number of classes in your classification task
    )
    data_dir_test = r'I:\0757 hyperspectral\RGB\final_side_dataset\test'

    test_dataset = datasets.ImageFolder(data_dir_test)
    test_dataset = TransformDataset(test_dataset, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.test_batch_size, shuffle=False)

    # 初始化模型、优化器和损失函数
    model_ft = FlexibleClassifier(str(architecture), num_classes)

    Best_F1 = 0

    patience = 20
    early_stopping = EarlyStopping(patience, verbose=True)


    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)

    txt = file_dir + '/' + 'config.txt'
    filename = open(txt, 'w')  # dict to txt
    for k, v in config.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()

    # criterion to GPU
    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()


    if resume:
        model_ft.load_state_dict(torch.load(model_path))

    model_ft.to(DEVICE)

    wandb.watch(model_ft, log="all")

    # if choose Adam, reduce learning rate
    if optimizer == 'SGD' or 'sgd':

        optimizer = sgd_optimizer(model_ft.named_parameters(), model_lr, momentum=wandb.config.opt_momentum,
                                  weight_decay=wandb.config.opt_weight_decay, use_custwd=False)

    elif optimizer == 'ADAM' or 'Adam':

        optimizer = adam_optimizer(model_ft.named_parameters(), model_lr, weight_decay=wandb.config.opt_weight_decay,
                                   use_custwd=False)

    elif optimizer == 'ADAM_normal' or 'Adam_normal':

        optimizer = optim.Adam(model_ft.named_parameters(), lr=model_lr)
    elif optimizer == 'SGD_normal' or 'sgd_normal':

        optimizer = optim.sgd(model_ft.named_parameters(), lr=model_lr)

    if schedule_lr == 'cosine':
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20,
                                                        eta_min=wandb.config.eta_min)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)

    if use_ema:
        ema = EMA(model_ft, 0.999)
        ema.register()


    total_save_epoch = []
    total_save_val_f1 = []
    total_save_test_f1 = []

    is_set_lr = False
    # train and val start
    for epoch in range(1, EPOCHS + 1):

        total_save_epoch.append(epoch)
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch)


        wandb.log({
            "Epoch": epoch,
            "Train Accuracy": train_acc,
            "Train Loss": train_loss
        })

        val_list, pred_list, val_loss, val_acc, F1 = val(model_ft, DEVICE, val_loader)

        total_save_val_f1.append(F1)

        if early_stopping and epoch >= 20:
            early_stopping(-F1, model_ft)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")

                print(classification_report(val_list, pred_list))
                # 生成混淆矩阵
                cm = confusion_matrix(val_list, pred_list)

                print("Confusion Matrix:")
                print(cm)

                wandb.log({
                    "Epoch": epoch,
                    "Val Accuracy": val_acc,
                    "Val Loss": val_loss,
                    'Val F1': F1
                })

                print('--------------------- testing ---------------------')

                val_list, pred_list, val_loss, test_acc, test_F1 = test(model_ft, DEVICE, test_loader)

                print(classification_report(val_list, pred_list))
                # 生成混淆矩阵
                cm = confusion_matrix(val_list, pred_list)

                print("Confusion Matrix:")
                print(cm)
                print('test F1 ' + str(test_F1))
                total_save_test_f1.append(test_F1)
                # 结束模型训练
                break

        print(classification_report(val_list, pred_list))
        # 生成混淆矩阵
        cm = confusion_matrix(val_list, pred_list)

        print("Confusion Matrix:")
        print(cm)

        wandb.log({
            "Epoch": epoch,
            "Val Accuracy": val_acc,
            "Val Loss": val_loss,
            'Val F1': F1
        })

        print('--------------------- testing ---------------------')

        val_list, pred_list, val_loss, test_acc, test_F1 = test(model_ft, DEVICE, test_loader)

        print(classification_report(val_list, pred_list))
        # 生成混淆矩阵
        cm = confusion_matrix(val_list, pred_list)

        print("Confusion Matrix:")
        print(cm)

        print('test F1 + ' + str(test_F1))

        total_save_test_f1.append(test_F1)


        if epoch < 100:
            schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = wandb.config.final_learning_rate
                    is_set_lr = True

    # 创建一个字典，其中包含所有数据
    data = {
        'Epoch': total_save_epoch,
        'Validation F1 Scores': total_save_val_f1,
        'Test F1 Scores': total_save_test_f1
    }

    # 使用pandas创建DataFrame
    df = pd.DataFrame(data)

    # 保存DataFrame到Excel文件
    filename = file_dir +'/' + 'f1_scores_per_epoch.xlsx'
    df.to_excel(filename, index=False)

    print(f'Data saved to Excel successfully: {filename}')

