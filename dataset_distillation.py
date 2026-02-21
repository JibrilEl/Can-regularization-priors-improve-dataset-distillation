import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


class TwoLayerMLP(nn.Module):

    def __init__(self):
        super(TwoLayerMLP, self).__init__()

        self.lin1 = nn.Linear(28*28, 128)
        self.lin2 = nn.Linear(128, 10)
    
    def forward(self, x):

        B, C, H, W = x.shape
        
        x = x.view(B, -1)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


def compute_gd_step(models, images, labels, eta):

    updated_models = []
    image_tensor = torch.stack(images)
    
    for i in range(len(models)):
        curr_model = models[i]
        curr_model.train()
        
        output = curr_model(image_tensor)
        loss = F.cross_entropy(output, labels)
        
        grads = torch.autograd.grad(
            loss, 
            curr_model.parameters(), 
            create_graph=True
        )
        
        updated_params = []
        for param, grad in zip(curr_model.parameters(), grads):
            updated_params.append(param - eta * grad)
        
        updated_models.append((curr_model, updated_params))
    
    return updated_models


def eval_obj_function(test_data, test_labels, updated_models):

    criterion = nn.CrossEntropyLoss()
    loss = 0
    
    for model, updated_params in updated_models:

        output = functional_forward(model, test_data, updated_params)
        loss += criterion(output, test_labels)
    
    return loss / len(updated_models)


def functional_forward(model, x, params):
    B, C, H, W = x.shape
    x = x.view(B, -1)
    
    x = F.linear(x, params[0], params[1])
    x = F.relu(x)
    
    x = F.linear(x, params[2], params[3])
    
    return x


def create_distilled_image(n_iter, dataloader, n_models, image_size, n_image, num_classes=10, alpha_images=1e2, alpha_eta=1e-2, prior="none", lambda_param=0.1, pretrained_path=None):
    
    images = []
    for _ in range(n_image):
        images.append(nn.Parameter(torch.randn(size=image_size, device=device)))
    
    if n_image % num_classes == 0:
        labels = torch.tensor([i for i in range(num_classes)] * (n_image//num_classes), device=device)
    else:
        raise ValueError("The number of images should be a multiple of the number of classes")
    
    eta = nn.Parameter(torch.tensor([2e0], device=device))
    eta_list = []
    losses_list = []
    
    if prior == "distill":
        pretrained_model = TwoLayerMLP()
        pretrained_model.load_state_dict(torch.load(pretrained_path, weights_only=False))
    
    optimizer_images = torch.optim.SGD(images, lr=alpha_images)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_images, n_iter)

    optimizer_eta = torch.optim.SGD([eta], lr=alpha_eta)

    best_loss, best_labels, best_images = torch.inf, None, None

    try:
        for k in range(n_iter):

            models = []
            for _ in range(n_models):
                mlp = TwoLayerMLP().to(device)
                models.append(mlp)

            optimizer_images.zero_grad()
            optimizer_eta.zero_grad()

            test_data, test_labels = next(iter(dataloader))

            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            updated_models = compute_gd_step(models, images, labels, eta)

            loss = eval_obj_function(test_data, test_labels, updated_models)

            if prior == "smoothness":
                for img in images:
                    loss += lambda_param * torch.linalg.norm(torch.flatten(img), ord=2)
            elif prior == "sparsity":
                for img in images:
                    loss += lambda_param * torch.linalg.norm(torch.flatten(img), ord=1)
            elif prior == "distill":
                for i in range(len(images)):
                    loss += lambda_param * nn.CrossEntropyLoss()(pretrained_model(images[i].unsqueeze(0)), torch.tensor([labels[i]])).sum()

            loss.backward()

            optimizer_images.step()
            scheduler.step()
            optimizer_eta.step()

            eta_list.append(eta.item())
            losses_list.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_images = images
                best_labels = labels

            if k % 20 == 1:
                print(f"Iter {k}, Loss: {loss.item():.4f}, Eta: {eta.item():.4f}, Scheduler LR : {scheduler.get_last_lr()[0]:.4f}")
    except KeyboardInterrupt:
        print(f"Interrupted manually at iteration : {k}")
        return best_images, best_labels, eta_list, losses_list
            
    return best_images, best_labels, eta_list, losses_list


def train_on_distilled_data(images, dataloader_test, learning_rate, n_epochs, val_freq, num_classes, n_image):

    val_accuracies = []
    if not isinstance(images, torch.Tensor):
        images = torch.stack(images)
    images = images.to(device)
    model = TwoLayerMLP().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    labels = torch.tensor([i for i in range(num_classes)] * (n_image//num_classes), device=device)
    for epoch in range(n_epochs):

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % val_freq == 0:
            val_loss = 0.0
            val_acc = 0
            for i, (data, label) in enumerate(dataloader_test):
            
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                val_loss += criterion(output, label).item()
    
                val_acc += (torch.argmax(output, -1) == label).sum().item()
    
            val_loss /= 10000
            val_acc /= 10000
    
            val_accuracies.append(val_acc)
    
            print(f"Epoch : {epoch}, val loss : {val_loss}, val acc : {val_acc}")
    
    return val_accuracies


def plot_loss_and_eta(losses_list, eta_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses_list)
    axes[1].plot(np.array(torch.tensor(eta_list).detach()))
    plt.tight_layout()
    plt.show()


def plot_images_single_row(images):
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))

    if len(images) == 1:
        axes = [axes]

    for i in range(len(images)):
        axes[i].imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_first_20_images(images):
    images_per_row = 5
    rows = (20 + images_per_row - 1) // images_per_row

    fig, axes = plt.subplots(rows, images_per_row, figsize=(3*images_per_row, 3*rows))
    axes = axes.flatten()

    for i in range(20):
        axes[i].imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0))
        axes[i].axis('off')
        axes[i].set_title(f"{i}", fontsize=8)

    for i in range(20, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def compute_and_plot_class_averages(images, labels, num_classes=10):
    images = torch.stack(images)
    images = images.squeeze()
    H, W = images.shape[1], images.shape[2]

    labels_expanded = labels.view(-1, 1, 1).expand(-1, H, W)

    summed_images = torch.zeros(num_classes, H, W, dtype=images.dtype, device=images.device)

    summed_images.scatter_add_(0, labels_expanded, images)

    counts = torch.bincount(labels, minlength=num_classes).view(num_classes, 1, 1).float()

    averaged_images = summed_images / counts

    print(f"Averaged images shape: {averaged_images.shape}")

    fig, axes = plt.subplots(1, len(averaged_images), figsize=(5*len(averaged_images), 5))

    if len(averaged_images) == 1:
        axes = [axes]

    for i in range(len(averaged_images)):
        axes[i].imshow(averaged_images[i].cpu().detach().numpy())
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return averaged_images


def main():
    parser = argparse.ArgumentParser(description="Dataset Distillation on MNIST")

    parser.add_argument("--prior", type=str, default="none", choices=["none", "smoothness", "sparsity", "distill"])
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--n_models", type=int, default=200)
    parser.add_argument("--n_image", type=int, default=10)
    parser.add_argument("--alpha_images", type=float, default=1e2)
    parser.add_argument("--alpha_eta", type=float, default=1e0)
    parser.add_argument("--lambda_param", type=float, default=0.01)
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_model_state_dict.pt")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--show_class_averages", action="store_true")

    args = parser.parse_args()

    dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.MNIST("./", train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1024, shuffle=True)

    images, labels, eta_list, losses_list = create_distilled_image(
        n_iter=args.n_iter,
        dataloader=dataloader,
        n_models=args.n_models,
        image_size=(1, 28, 28),
        n_image=args.n_image,
        num_classes=args.num_classes,
        alpha_images=args.alpha_images,
        alpha_eta=args.alpha_eta,
        prior=args.prior,
        lambda_param=args.lambda_param,
        pretrained_path=args.pretrained_path
    )

    plot_loss_and_eta(losses_list, eta_list)

    if args.n_image <= 10:
        plot_images_single_row(images)
    else:
        plot_first_20_images(images)

    if args.show_class_averages and args.n_image > 10:
        compute_and_plot_class_averages(images, labels, args.num_classes)

    val_accuracies = train_on_distilled_data(
        images=images,
        dataloader_test=dataloader_test,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        val_freq=args.val_freq,
        num_classes=args.num_classes,
        n_image=args.n_image
    )

    plt.plot(val_accuracies)
    plt.show()


if __name__ == "__main__":
    main()
