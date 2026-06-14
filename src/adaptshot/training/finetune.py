"""CA-EWC: Correction-Aware Elastic Weight Consolidation fine-tuning module.

Implements head-only continual learning that prevents catastrophic forgetting
of previously learned classes while adapting to new domain corrections.
Fisher Information diagonal computation is weighted by human feedback confidence.
"""

import logging
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover – torch is optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CAEWCFinetuner:
    """
    Correction-Aware Head-Only Fine-Tuning via Fisher Information regularization.

    IMPORTANT SCOPE NOTE (v0.2.0): This fine-tuner operates ONLY on the
    classification head — a single nn.Linear(embedding_dim, n_classes) layer
    containing ~(embedding_dim * n_classes) parameters (e.g., 2560 for 5-way
    with ResNet-18's 512-dim embeddings). It does NOT fine-tune the frozen
    backbone (ResNet/MobileNet). The term "Elastic Weight Consolidation"
    here refers to the Fisher-weighted regularization applied to these ~2K
    head parameters, not a full-network EWC implementation.

    For full backbone fine-tuning, use a dedicated training pipeline with
    GPU acceleration; this head-only approach is intentionally lightweight
    for CPU-first, resource-constrained environments.

    The penalty strength is modulated by the confidence weight of human corrections:
    - High confidence (1.0) -> Reduced penalty (Model adapts freely to the strong signal)
    - Low confidence (0.0) -> Full penalty (Model stays conservative to preserve existing knowledge)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ewc_lambda: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 16,
    ) -> None:
        """
        Args:
            model: The classification head or full model to fine-tune
            device: Target device (cpu/cuda)
            ewc_lambda: Weight of the EWC penalty term
            learning_rate: Optimizer learning rate
            epochs: Number of fine-tuning epochs per correction batch
            batch_size: Batch size for fine-tuning
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for fine-tuning. "
                "Install with: pip install 'adaptshot[torch]'"
            )
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        # State to store Fisher Information and old parameters
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._old_params: Optional[Dict[str, torch.Tensor]] = None

    def update_fisher(self, data_loader: DataLoader[Any]) -> Dict[str, torch.Tensor]:
        """
        Approximate diagonal Fisher Information Matrix for the model parameters.
        This should be called on the support set (old knowledge) before fine-tuning.

        Args:
            data_loader: DataLoader with representative support data

        Returns:
            Dict mapping parameter names to their diagonal Fisher tensors
        """
        self.model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        
        # Initialize fisher dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        # Compute gradients squared
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            
            output = self.model(inputs)
            loss = F.cross_entropy(output, targets)
            loss.backward()  # type: ignore[no-untyped-call]

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2)

        self._fisher = fisher
        
        # Snapshot current parameters
        self._old_params = {name: param.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}
        
        logger.debug(f"Fisher information updated for {len(fisher)} parameters.")
        return fisher

    def finetune(
        self, 
        new_embeddings: torch.Tensor, 
        new_labels: torch.Tensor, 
        confidence_weights: Optional[torch.Tensor] = None
    ) -> None:
        """
        Fine-tune the model on new embeddings with CA-EWC penalty.

        Args:
            new_embeddings: [N, D] tensor of new embeddings from corrections
            new_labels: [N] tensor of corrected labels (integers)
            confidence_weights: [N] tensor of human confidence scores [0, 1]
        """
        if self._fisher is None or self._old_params is None:
            logger.warning("Fisher not computed yet. Running standard fine-tuning without EWC penalty.")
            self._standard_finetune(new_embeddings, new_labels)
            return

        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Create DataLoader for new data
        if confidence_weights is None:
            confidence_weights = torch.ones(new_embeddings.size(0))
            
        dataset = TensorDataset(new_embeddings, new_labels, confidence_weights)
        loader: DataLoader[Any] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in loader:
                inputs, targets, weights = [x.to(self.device) for x in batch]
                optimizer.zero_grad()

                # Task Loss (Cross Entropy)
                output = self.model(inputs)
                task_loss = F.cross_entropy(output, targets)

                # EWC Loss
                ewc_loss = torch.tensor(0.0, device=self.device)
                for name, param in self.model.named_parameters():
                    if name in self._fisher and name in self._old_params:
                        # Scale penalty by (1 - confidence_weight)
                        # If confidence is high (1.0), penalty is 0 -> Model learns freely
                        # If confidence is low (0.0), penalty is 1 -> Model is conservative
                        weight_factor = (1.0 - weights.mean()).item()
                        ewc_loss += torch.sum(self._fisher[name] * (param - self._old_params[name]) ** 2) * weight_factor

                total_loss = task_loss + self.ewc_lambda * ewc_loss
                total_loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += total_loss.item()

        self.model.eval()
        logger.info(f"CA-EWC Finetuning complete after {self.epochs} epochs.")

    def _standard_finetune(self, new_embeddings: torch.Tensor, new_labels: torch.Tensor) -> None:
        """Standard fine-tuning without EWC."""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        dataset = TensorDataset(new_embeddings, new_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for _ in range(self.epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = F.cross_entropy(output, targets)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
        
        self.model.eval()
