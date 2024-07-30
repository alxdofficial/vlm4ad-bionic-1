import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLayer(nn.Module):
    def __init__(self, dim, batch_size, alpha_init=0.1, decay_init=0.01, device='cuda'):
        super(HebbianLayer, self).__init__()
        self.dim = dim
        self.device = device
        
        # Weights and recurrent weights
        self.weights = nn.Parameter(torch.randn(dim, dim, device=self.device) * 0.01)
        self.recurrent_weights = nn.Parameter(torch.randn(dim, dim, device=self.device) * 0.01)
        
        # Hebbian parameters
        self.alpha = nn.Parameter(torch.full((dim,), alpha_init, device=self.device))
        self.decay = nn.Parameter(torch.full((dim,), decay_init, device=self.device))
        
        # Neuromodulators
        self.dopamine = torch.zeros(batch_size, dim, device=self.device)
        self.serotonin = torch.zeros(batch_size, dim, device=self.device)
        self.glutamate = torch.zeros(batch_size, dim, device=self.device)
        self.gaba = torch.zeros(batch_size, dim, device=self.device)
        
        # Store previous activations
        self.previous_activation = torch.zeros(batch_size, dim, device=self.device)
        self.previous_recurrent_activation = torch.zeros(batch_size, dim, device=self.device)

        # Neural network to predict neurotransmitters
        self.neurotransmitter_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 4 * dim),  # 4 for dopamine, serotonin, glutamate, gaba
            nn.Tanh()  # Ensures outputs are within the range [-1, 1]
        )

    def forward_(self, x):
        batch_size = x.size(0)

        # Compute recurrent input
        recurrent_input = torch.matmul(self.previous_recurrent_activation, self.recurrent_weights)  # (batch_size, dim)

        # Compute activations
        activations = F.relu(torch.matmul(x, self.weights) + recurrent_input)

        # Apply thresholds based on glutamate and GABA
        threshold = 0.5 * (1 - self.gaba + self.glutamate)  # (batch_size, dim)
        activations = F.relu(activations - threshold)

        # Update previous activations
        self.previous_activation = activations.detach()
        self.previous_recurrent_activation = recurrent_input.detach()

        return activations

    def hebbian_update(self):
        def calculate_hebbian_updates(previous_activation, weights):
            # Calculate softmax over activations and weights
            activation_softmax = F.softmax(previous_activation, dim=1)
            weight_softmax = F.softmax(weights, dim=1)

            # Calculate Hebbian updates
            hebbian_updates = activation_softmax.unsqueeze(2) * weight_softmax.unsqueeze(0)
            return hebbian_updates

        # Compute Hebbian updates for both regular and recurrent weights
        hebbian_updates = calculate_hebbian_updates(self.previous_activation, self.weights)
        recurrent_hebbian_updates = calculate_hebbian_updates(self.previous_recurrent_activation, self.recurrent_weights)

        # Modulate updates with alpha and dopamine
        modulated_alpha = self.alpha.unsqueeze(0) * self.dopamine * self.serotonin  # (batch_size, dim)
        hebbian_updates *= modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)
        recurrent_hebbian_updates *= modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)

        # Average over the batch dimension and apply updates to weights
        weight_update = hebbian_updates.mean(dim=0)  # (dim, dim)
        recurrent_weight_update = recurrent_hebbian_updates.mean(dim=0)  # (dim, dim)

        # Apply updates to weights and recurrent weights
        updated_weights = self.weights + weight_update
        updated_recurrent_weights = self.recurrent_weights + recurrent_weight_update

        # Apply neuron-specific decay to weights, ensuring they decay towards zero
        decay_matrix = torch.diag(self.decay).to(self.device)
        updated_weights -= torch.matmul(decay_matrix, self.weights)
        updated_recurrent_weights -= torch.matmul(decay_matrix, self.recurrent_weights)

        self.weights.data.copy_(updated_weights)
        self.recurrent_weights.data.copy_(updated_recurrent_weights)

    def set_neuromodulators(self):
        # Predict new neurotransmitter levels using the previous activation
        neurotransmitter_output = self.neurotransmitter_predictor(self.previous_activation)
        neurotransmitter_output = neurotransmitter_output.view(-1, self.dim, 4)  # (batch_size, dim, 4)

        # Extract neurotransmitter predictions
        pred_dopamine, pred_serotonin, pred_glutamate, pred_gaba = neurotransmitter_output.unbind(dim=2)

        # Calculate inverse scale based on serotonin magnitude
        serotonin_magnitude = torch.abs(pred_serotonin).clamp(min=1e-6)  # Avoid division by zero
        inverse_scale = 1 / serotonin_magnitude

        # Update neuromodulators with new values; the inputs are updates
        self.dopamine = self.dopamine + pred_dopamine * inverse_scale
        self.serotonin = self.serotonin + pred_serotonin
        self.glutamate = self.glutamate + pred_glutamate * inverse_scale
        self.gaba = self.gaba + pred_gaba * inverse_scale

    def forward(self, x):
        res = self.forward_(x)
        self.hebbian_update()
        self.set_neuromodulators()
    
    def get_non_trainable(self):
        # Returns the names of the weight and recurrent weight parameters
        return ['weights', 'recurrent_weights']
    




# Dummy data and targets for testing
def generate_dummy_data(batch_size, dim):
    return torch.randn(batch_size, dim), torch.randint(0, 2, (batch_size, dim)).float()

# Example usage
batch_size = 4
dim = 10
epochs = 20

hebbian_layer = HebbianLayer(dim, batch_size)
optimizer = torch.optim.SGD(hebbian_layer.parameters(), lr=0.01)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Generate dummy data and targets
    x, target = generate_dummy_data(batch_size, dim)
    output = hebbian_layer(x)
    
    # Compute loss (e.g., Mean Squared Error)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Set gradients of non-trainable parameters to zero
    for name, param in hebbian_layer.named_parameters():
        if name in hebbian_layer.get_non_trainable():
            param.grad = None
    
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")