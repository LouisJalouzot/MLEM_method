from src.utils import device
from joblib.memory import Memory

memory = Memory(location=".cache", verbose=0)


@memory.cache
def compute_embeddings(model, dataloader):
    """
    Computes the embeddings of a model on a dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): The data loader.

    Returns:
        torch.Tensor: The computed embeddings.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings.append(model(batch).detach().cpu())
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings
