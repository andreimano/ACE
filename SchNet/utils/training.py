import torch
import torch.nn.functional as F

def get_loss(data, pred):
    return F.l1_loss(pred.view(-1), data.y), "MAE"


def get_metrics(data, pred):
    return (pred.view(-1) - data.y).abs(), "MAE"


def train(
    model,
    loader,
    optimizer,
    optimizer_dual,
    lambdas,
    gammas,
    device,
    args,
    constrained=False,
    disable_equivariance=False,
):
    model.train()
    total_loss = 0

    if args.n_gammas == "nn":
        all_gammas = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.n_gammas == "nn":
            pred, gammas = model(data.z, data.pos, data.batch, disable_equivariance=disable_equivariance)
            all_gammas.append(torch.stack(gammas).unsqueeze(0))
        else:
            pred = model(data.z, data.pos, data.batch, disable_equivariance=disable_equivariance)

        loss, loss_name = get_loss(data, pred)

        if constrained:
            slacks = [gamma for gamma in gammas]
            dual_loss = 0

            for dual_var, slack in zip(lambdas, slacks):
                dual_loss += dual_var * slack + args.auglag_const * slack ** 2

            loss += dual_loss

            for ii, slack in enumerate(slacks):
                lambdas[ii].grad = None if slack is None else -slack

        loss.backward()
        optimizer.step()

        if constrained:
            optimizer_dual.step()

        total_loss += loss.item() * data.num_graphs

    if args.n_gammas == "nn":
        all_gammas = torch.cat(all_gammas, dim=0)
        gammas = all_gammas.mean(dim=0)
        return total_loss / len(loader.dataset), loss_name, gammas

    return total_loss / len(loader.dataset), loss_name, None


@torch.no_grad()
def evaluate(model, loader, device, args, disable_equivariance=False):
    model.eval()
    metrics = []
    all_gammas = []
    for data in loader:
        data = data.to(device)
        if args.n_gammas == "nn":
            pred, gammas = model(data.z, data.pos, data.batch, disable_equivariance=disable_equivariance)
        else:
            pred = model(data.z, data.pos, data.batch, disable_equivariance=disable_equivariance)

        if args.n_gammas == "nn":
            all_gammas.append(torch.stack(gammas).unsqueeze(0))

        metric, metric_name = get_metrics(data, pred)

        metrics.append(metric)

    metric = torch.cat(metrics, dim=0)

    if args.n_gammas == "nn":
        all_gammas = torch.cat(all_gammas, dim=0)
        gammas = all_gammas.mean(dim=0)
        return metric.mean().item(), metric.std().item(), metric_name, gammas

    return metric.mean().item(), metric.std().item(), metric_name, None
