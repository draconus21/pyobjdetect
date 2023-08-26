import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os
from pathlib import Path
from PIL import Image
from tempfile import TemporaryDirectory

from pyobjdetect.utils import misc, helpers, logutils, viz


cudnn.benchmark = True

BEST_MODEL_PARAMS_PATH = "best_model_params.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_dir():
    return os.path.join(misc.get_data_dir(), "hymenoptera_data")


def get_transforms():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        ),
    }

    return data_transforms


def get_datasets():
    data_dir = get_data_dir()
    data_transforms = get_transforms()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

    return image_datasets


def previz(inp):
    inp = helpers.torch2numpy(inp)
    inp = STD * inp + MEAN
    inp = np.clip(inp, 0, 1)
    return inp


def train_one_epoch(dataloaders, dataset_sizes, model, criterion, optimizer, device, phase):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameters gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            # backwards + optimize only if in training phase
            if phase == "train":
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    logutils.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def train_model(
    dataloaders, dataset_sizes, class_names, model, criterion, optimizer, scheduler, device, num_epochs=25
):
    since = time.time()
    logutils.info("Training starting")

    logutils.debug(f"classes in dataset: {class_names}")

    logutils.debug(f"Running on {device}")

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as traindir:
        logutils.debug(f"Training dir: {Path(traindir).absolute()}")

        best_model_params_path = Path(os.path.join(traindir, BEST_MODEL_PARAMS_PATH)).absolute()

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            logutils.info(f"Epoch {epoch+1}/{num_epochs}")
            logutils.secho("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set to train model
                else:
                    model.eval()  # Set to evaluation model

                # Iterate over data
                epoch_loss, epoch_acc = train_one_epoch(
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    phase=phase,
                )

                # visualize_model(
                #    model=model,
                #    dataloader=dataloaders[phase],
                #    class_names=class_names,
                #    device=device,
                #    title=f"epoch: {epoch+1}",
                # )

                if phase == "train":
                    scheduler.step()

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        logutils.info(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        logutils.info(f"Best val Acc: {best_acc:.4f}")
        logutils.info(f"Best model saved at {best_model_params_path}")

        model.load_state_dict(torch.load(best_model_params_path))

    return model


def visualize_model(model, dataloader, class_names, num_images=6, device="cpu", title=None):
    was_training = model.training
    model.eval()
    images_so_far = 0

    fig, ax, _, _ = viz.subplots_n(num_images, title=title)
    ax = ax.ravel()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                viz.matshow(
                    ax[images_so_far],
                    previz(inputs.cpu().data[j]),
                    title=f"prediction: {class_names[preds[j]]}",
                    cbar=False,
                    show_axis=False,
                )

                images_so_far += 1
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def run(**kwargs):
    logutils.setupLogging(level="DEBUG")

    # data params
    bsize = kwargs.get("batch_size", 4)
    shuffle = kwargs.get("shuffle", True)
    num_workers = kwargs.get("num_workers", 4)
    gpu_num = kwargs.get("gpu_num", 0)

    # train params
    num_epochs = kwargs.get("num_epochs", 25)
    lr = kwargs.get("lr", 0.001)
    momentum = kwargs.get("momentum", 0.9)
    lr_step_size = kwargs.get("lr_step_size", 7)
    lr_gamma = kwargs.get("lr_gamma", 0.1)

    image_datasets = get_datasets()

    dataset_keys = [k for k in image_datasets.keys()]

    class_names = image_datasets[dataset_keys[0]].classes
    logutils.debug(f"classes in dataset: {class_names}")

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bsize, shuffle=shuffle, num_workers=num_workers)
        for x in dataset_keys
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_keys}
    logutils.debug(f"dataset size: {dataset_sizes}")

    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    logutils.debug(f"Running on {device}")

    # get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))

    # make a grid
    out = [previz(inputs[i]) for i in range(len(inputs))]

    # display them
    viz.quickmatshow(out, title="Train sample", cbar=False, show_axis=False)

    ###############################
    # Preparing pre trained model #
    ###############################

    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    numt_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to `num_classes`
    model_ft.fc = nn.Linear(numt_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=lr_step_size, gamma=lr_gamma)

    model_ft = train_model(
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        class_names=class_names,
        model=model_ft,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        device=device,
        num_epochs=num_epochs,
    )

    visualize_model(
        model=model_ft,
        dataloader=dataloaders["val"],
        class_names=class_names,
        num_images=6,
        device=device,
        title="Best Model",
    )

    viz.show()


if __name__ == "__main__":
    run()
