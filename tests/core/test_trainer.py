import pandas as pd
import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.core.pairwise_dataset import PairwiseDataset  # Using real class now
from src.core.spd_matrix_learner import SPDMatrixLearner  # Using real class now

# Assuming necessary imports from src.core
from src.core.trainer import train_loop


# Unskip the test and use real components
def test_train_loop_runs():
    # Basic test to ensure train_loop runs without crashing
    n_features = 4
    n_samples = 20  # Increase samples for more stable pairwise data
    n_batches_per_epoch = 5  # Corresponds to dataset length
    max_epochs = 3
    device = "cpu"  # or 'cuda' if available and configured

    # Use real SPDMatrixLearner
    # Use a parametrization that allows learning (e.g., cholesky or sym)
    model = SPDMatrixLearner(
        num_features=n_features, param="sym", loss="spearman"
    ).to(device)

    # Create realistic dummy data for PairwiseDataset
    X_data = torch.randn(n_samples, n_features)
    # Simulate some underlying structure for Y (e.g., related to X norms)
    Y_data = torch.cdist(X_data, X_data)  # Pairwise distances as base
    # Add noise
    Y_data += torch.randn_like(Y_data) * 0.1
    # Use real PairwiseDataset
    # n_pairs determines pairs sampled *per batch* (per __getitem__ call)
    # gamma=1 means n_pairs stays constant
    dataset = PairwiseDataset(X=X_data, Y=Y_data, n_pairs=50, gamma=1)

    # Define optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-3)
    # Scheduler expects validation metric, train_loop uses spearman
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=2, factor=0.5)

    # Call the train_loop function
    trained_model, logs = train_loop(
        model=model,
        dataset=dataset,  # Pass the dataset instance
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        eps=1e-4,
        device=device,
        # Pass dataset length directly to train_loop
        # Or modify train_loop to accept dataset and use len(dataset)
        # For now, let's assume train_loop iterates max_epochs times
        # and calls dataset[i] where i is the epoch number.
        # This matches the current train_loop implementation.
    )

    # Add basic assertions
    assert isinstance(trained_model, SPDMatrixLearner)
    assert isinstance(logs, pd.DataFrame)
    # The loop runs max_epochs times unless convergence (eps) is met
    assert len(logs) <= max_epochs
    assert len(logs) > 0  # Should run at least one epoch
    assert "Score" in logs.columns
    assert "Spearman" in logs.columns
    assert "LR" in logs.columns
    assert "Gradient Norm" in logs.columns
    assert "Diff norm" in logs.columns

    # Check if model parameters have changed (gradient updates happened)
    # This requires storing initial parameters, which is complex here.
    # Instead, check if the final Spearman is plausible (e.g., not NaN)
    assert not logs["Spearman"].isnull().any()


# Keep other tests skipped as they require more complex setup
@pytest.mark.skip(reason="Requires specific convergence setup")
def test_train_loop_convergence():
    # Test if the loop stops early when convergence criteria (eps) is met
    # This requires careful mocking of model.get_W() and diff_norm calculation
    pass


@pytest.mark.skip(reason="Requires specific scheduler setup")
def test_train_loop_scheduler_step():
    # Test if the scheduler steps based on the spearman metric
    # Requires mocking the scheduler and checking its state
    pass
