import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLayer(nn.Module):
    def __init__(self, dim, batch_size, alpha_init=0.1, decay_init=0.01, device='cuda'):
        super(HebbianLayer, self).__init__()
        self.dim = dim
        self.device = device
        
        # Hebbian weights and recurrent weights
        self.hebbian_weights = nn.Parameter(torch.randn(dim, dim, device=device) * 0.01)
        self.hebbian_recurrent_weights = nn.Parameter(torch.randn(dim, dim, device=device) * 0.01)
        
        # Hebbian parameters
        self.alpha = nn.Parameter(torch.full((dim,), alpha_init, device=device))
        self.decay = nn.Parameter(torch.full((dim,), decay_init, device=device))
        
        # Neuromodulators
        self.dopamine = torch.zeros(batch_size, dim, device=device)
        self.serotonin = torch.zeros(batch_size, dim, device=device)
        self.gaba = torch.zeros(batch_size, dim, device=device)

        # Layer normalization layers
        self.layer_norm_activations = nn.LayerNorm(dim)
        self.layer_norm_recurrent = nn.LayerNorm(dim)

        # Neural network to predict neurotransmitters
        self.neurotransmitter_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3 * dim),  # 3 for dopamine, serotonin, gaba
        )

        # Store previous activations
        self.previous_activation = torch.zeros(batch_size, dim, device=device)
        self.previous_recurrent_activation = torch.zeros(batch_size, dim, device=device)

    def forward(self, x):
        # Compute recurrent input
        recurrent_input = torch.matmul(self.previous_activation, self.hebbian_recurrent_weights)  # (batch_size, dim)
        recurrent_input = self.layer_norm_recurrent(recurrent_input)
        self.previous_recurrent_activation = recurrent_input.detach()
        
        # Compute activations
        activations = F.relu(torch.matmul(x, self.hebbian_weights) + recurrent_input)
        activations = self.layer_norm_activations(activations)
        self.previous_activation = activations.detach()

        self.update_neuromodulators()
        self.hebbian_update()
    
        return activations

    def hebbian_update(self):
        def calculate_hebbian_updates(prev_activation, weights):
            # Calculate softmax over activations and weights
            activation_softmax = F.softmax(prev_activation, dim=1)
            weight_softmax = F.softmax(weights, dim=1)

            # Calculate Hebbian updates
            hebbian_updates = activation_softmax.unsqueeze(2) * weight_softmax.unsqueeze(0)
            return hebbian_updates

        # Compute Hebbian updates for both regular and recurrent weights
        hebbian_updates = calculate_hebbian_updates(self.previous_activation, self.hebbian_weights)
        recurrent_hebbian_updates = calculate_hebbian_updates(self.previous_recurrent_activation, self.hebbian_recurrent_weights)

        # Modulate updates with alpha and dopamine
        modulated_alpha = self.alpha.unsqueeze(0) * self.dopamine * self.serotonin  # (batch_size, dim)
        hebbian_updates = hebbian_updates * modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)
        recurrent_hebbian_updates = recurrent_hebbian_updates * modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)

        # Average over the batch dimension and apply updates to weights
        weight_update = hebbian_updates.mean(dim=0)  # (dim, dim)
        recurrent_weight_update = recurrent_hebbian_updates.mean(dim=0)  # (dim, dim)

        # Apply updates to weights and recurrent weights
        updated_weights = self.hebbian_weights + weight_update
        updated_recurrent_weights = self.hebbian_recurrent_weights + recurrent_weight_update

        # Apply neuron-specific decay to weights, scaled by GABA, ensuring they decay towards zero
        scaled_decay = self.decay * torch.sigmoid(self.gaba.mean(dim=0))
        decay_matrix = torch.diag(scaled_decay)
        updated_weights = updated_weights - torch.matmul(decay_matrix, self.hebbian_weights)
        updated_recurrent_weights = updated_recurrent_weights - torch.matmul(decay_matrix, self.hebbian_recurrent_weights)

        self.hebbian_weights.data.copy_(updated_weights)
        self.hebbian_recurrent_weights.data.copy_(updated_recurrent_weights)

    def update_neuromodulators(self):
        # Predict new neurotransmitter updates using the previous activation
        neurotransmitter_output = self.neurotransmitter_predictor(self.previous_activation)
        neurotransmitter_output = neurotransmitter_output.view(-1, self.dim, 3)  # (batch_size, dim, 3)

        # Extract neurotransmitter predictions
        pred_dopamine, pred_serotonin, pred_gaba = neurotransmitter_output.unbind(dim=2)

        # Calculate inverse scale based on serotonin magnitude
        serotonin_magnitude = pred_serotonin.clamp(min=1e-6)  # Avoid division by zero
        inverse_scale = 1 / serotonin_magnitude

        # Detach the current neuromodulator values before updating them
        self.dopamine = torch.tanh(self.dopamine.detach() + pred_dopamine * inverse_scale)
        self.serotonin = torch.sigmoid(self.serotonin.detach() + pred_serotonin)
        self.gaba = torch.sigmoid(self.gaba.detach() + pred_gaba * inverse_scale)
    
    def get_non_trainable(self):
        # Returns the names of the weight and recurrent weight parameters
        return ['hebbian_weights', 'hebbian_recurrent_weights']

# Dummy data and targets for testing
def generate_dummy_data(batch_size, dim):
    return torch.randn(batch_size, dim), torch.randint(0, 2, (batch_size, dim)).float()

# Example usage
batch_size = 4
dim = 10
epochs = 20

hebbian_layer = HebbianLayer(dim, batch_size).to('cuda')
optimizer = torch.optim.SGD(hebbian_layer.parameters(), lr=0.01)

torch.autograd.set_detect_anomaly(True)
# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Generate dummy data and targets
    x, target = generate_dummy_data(batch_size, dim)
    x = x.to('cuda')
    target = target.to('cuda')
    activations = hebbian_layer(x)
    
    # Compute loss (e.g., Mean Squared Error)
    loss = F.mse_loss(activations, target)
    loss.backward()
    
    # Set gradients of non-trainable parameters to zero
    for name, param in hebbian_layer.named_parameters():
        if name in hebbian_layer.get_non_trainable():
            param.grad = None
    
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
