import tqdm
from src.utils import metrics as utils
from src.analysis.stats import compute_statistics

def train_one_epoch(model, dataloader, optimizer, criterion, device, 
                    optimizer_name, results, prev_params, 
                    log_interval=10, compute_stats=False):
    """
    Trains for one epoch. 
    Returns: avg_loss, accuracy, updated_prev_params, updated_results
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm.tqdm(dataloader, desc=f"Train {optimizer_name}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Capture gradient before step if we need stats
        flat_grad = None
        if compute_stats and (batch_idx + 1) % log_interval == 0:
            flat_grad = utils.flatten_grad(model)

        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Compute Heavy Statistics
        if compute_stats and flat_grad is not None:
            # We assume batch size is constant or handle it inside
            prev_params, results = compute_statistics(
                results=results,
                net=model,
                criterion=criterion,
                input_img=data, # PyHessian handles reshaping
                label=target,
                num_batch=data.size(0),
                prev_params=prev_params,
                flat_grad=flat_grad,
                optimizer_key=optimizer_name
            )

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, prev_params, results
