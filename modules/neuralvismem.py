import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the input and output dimensions for internal modules
INTERNAL_DIM = 768
NUM_SUB_NETWORKS = 2
NUM_LAYERS = 2
NUM_MEMORY_NETWORKS = 8
IMG_WIDTH = 1600
IMG_HEIGHT = 900

# 1. Feature Extractor
# The FeatureExtractor class consists of convolutional layers with batch normalization, ReLU activation, and dropout for feature extraction from images.
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, num_layers=5, kernel_size=5, stride=3, dropout=0.3):
        super(FeatureExtractor, self).__init__()
        channels = [16, 64, 128, 256, 512]  # Specified number of channels for each layer
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, channels[i], kernel_size=kernel_size, stride=stride))
            if i != num_layers - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_channels = channels[i]
        self.extractor = nn.Sequential(*layers)

    def forward(self, x):   
        return self.extractor(x)
# 2. Fully Connected Layer
# The FullyConnected class creates a fully connected network with ReLU activation and dropout.
class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(FullyConnected, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# takes in an embedding of peripheral vision and one cls token, predicts x and y coordinate of fovea as percent
class FoveaPosPredictor(nn.Module):
    def __init__(self, dropout=0.3):
        super(FoveaPosPredictor, self).__init__()
        hidden_dims = [256, 128, 32]
        self.fc = FullyConnected(INTERNAL_DIM + INTERNAL_DIM, hidden_dims, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, peripheral_embedding, cls_token):
        # Concatenate peripheral image encoding and CLS token along the last dimension
        # print(f"in fovea pos pred forward, periph embedding shape: ", peripheral_embedding.shape)
        # print(f"in fovea pos pred forward, cls token shape: ", cls_token.shape)
        combined_input = torch.cat((peripheral_embedding, cls_token), dim=-1)
        # Pass the concatenated tensor through the fully connected layers
        output = self.fc(combined_input)
        # Apply sigmoid activation to ensure the output values are between 0 and 1
        output = self.sigmoid(output)

        return output

# 3. Vision Encoder
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.img_width = IMG_WIDTH
        self.img_height = IMG_HEIGHT

        # Feature extractors
        self.fovea_feature_extractor = FeatureExtractor()
        self.peripheral_feature_extractor = FeatureExtractor()
        
        # Fully connected layers
        flattened_dim = 512  # Output from feature extractor after flattening
        hidden_dims = [256]
        output_dim = INTERNAL_DIM
        
        self.fovea_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)
        self.peripheral_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)

    def forward_fovea(self, images, fovea_coords):
        # images: (batch_size, 3, img_height, img_width)
        # fovea_coords: (batch_size, 2) - (x, y) in percentage

        batch_size = images.size(0)
        x_coords = (fovea_coords[:, 0] * self.img_width).int()
        y_coords = (fovea_coords[:, 1] * self.img_height).int()

        # Crop the fovea image (centered around the scaled coordinates in the original image)
        crop_size = 512
        half_crop = crop_size // 2

        fovea_images = []
        for i in range(batch_size):
            x, y = x_coords[i], y_coords[i]
            x = min(max(x, half_crop), self.img_width - half_crop)
            y = min(max(y, half_crop), self.img_height - half_crop)
            fovea_image = images[i:i+1, :, y-half_crop:y+half_crop, x-half_crop:x+half_crop]
            # fovea_image: (1, 3, 512, 512)
            fovea_images.append(fovea_image)

        fovea_images = torch.cat(fovea_images, dim=0)
        # fovea_images: (batch_size, 3, 512, 512)

        # Fovea feature extraction
        fovea_features = self.fovea_feature_extractor(fovea_images)
        # fovea_features: (batch_size, channels, h, w) after feature extraction 
        fovea_features = fovea_features.view(fovea_features.size(0), -1)  # Flatten
        # fovea_features: (batch_size, flattened_dim)
        fovea_output = self.fovea_fc(fovea_features)
        # fovea_output: (batch_size, output_dim)

        return fovea_output

    def forward_peripheral(self, image):
        # image: (batch_size, 3, original_height, original_width)
        # print(f"in vision encoder forward_peripheral, image shape: ", image.shape)
        # Resize the entire image
        peripheral_image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        # peripheral_image: (batch_size, 3, 512, 512)

        # Peripheral feature extraction
        peripheral_features = self.peripheral_feature_extractor(peripheral_image)
        # peripheral_features: (batch_size, channels, h, w) after feature extraction, h,w is 1
        peripheral_features = peripheral_features.view(peripheral_features.size(0), -1)  # Flatten
        # peripheral_features: (batch_size, flattened_dim)
        peripheral_output = self.peripheral_fc(peripheral_features)
        # peripheral_output: (batch_size, output_dim)

        return peripheral_output

# 4. Hebbian Layer
# The HebbianLayer class defines a layer with Hebbian learning, where weights are updated based on the activity of neurons and the presence of a neuro modulators.    
class HebbianLayer(nn.Module):
    def __init__(self, dim, alpha_init=0.1, decay_init=0.01, device='cuda'):
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
        self.dopamine = None
        self.serotonin = None
        self.gaba = None

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
        self.previous_activation = None
        self.previous_recurrent_activation = None

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize previous activations and neuromodulators if they are not set
        if self.previous_activation is None:
            self.previous_activation = torch.zeros(batch_size, self.dim, device=self.device)
            self.previous_recurrent_activation = torch.zeros(batch_size, self.dim, device=self.device)
            self.dopamine = torch.zeros(batch_size, self.dim, device=self.device)
            self.serotonin = torch.zeros(batch_size, self.dim, device=self.device)
            self.gaba = torch.zeros(batch_size, self.dim, device=self.device)

        self.update_neuromodulators()
        self.hebbian_update()

        # Compute recurrent input
        recurrent_input = torch.matmul(self.previous_activation, self.hebbian_recurrent_weights)  # (batch_size, dim)
        recurrent_input = self.layer_norm_recurrent(recurrent_input)
        self.previous_recurrent_activation = recurrent_input.detach()
        
        # Compute activations
        activations = F.relu(torch.matmul(x, self.hebbian_weights) + recurrent_input)
        activations = self.layer_norm_activations(activations)
        self.previous_activation = activations.detach()


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
        hebbian_updates = calculate_hebbian_updates(self.previous_activation, self.hebbian_weights.detach())
        recurrent_hebbian_updates = calculate_hebbian_updates(self.previous_recurrent_activation, self.hebbian_recurrent_weights.detach())

        # Modulate updates with alpha and dopamine
        modulated_alpha = self.alpha.unsqueeze(0) * self.dopamine * self.serotonin  # (batch_size, dim)
        hebbian_updates = hebbian_updates * modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)
        recurrent_hebbian_updates = recurrent_hebbian_updates * modulated_alpha.unsqueeze(1)  # (batch_size, dim, dim) * (batch_size, dim, 1)

        # Average over the batch dimension and apply updates to weights
        weight_update = hebbian_updates.mean(dim=0)  # (dim, dim)
        recurrent_weight_update = recurrent_hebbian_updates.mean(dim=0)  # (dim, dim)

        # Apply updates to weights and recurrent weights
        updated_weights = self.hebbian_weights.detach() + weight_update
        updated_recurrent_weights = self.hebbian_recurrent_weights.detach() + recurrent_weight_update

        # Apply neuron-specific decay to weights, scaled by GABA, ensuring they decay towards zero
        scaled_decay = self.decay * torch.sigmoid(self.gaba.mean(dim=0))
        decay_matrix = torch.diag(scaled_decay)
        updated_weights = updated_weights - torch.matmul(decay_matrix, self.hebbian_weights.detach())
        updated_recurrent_weights = updated_recurrent_weights - torch.matmul(decay_matrix, self.hebbian_recurrent_weights.detach())

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
    


# 5. SubNetwork
# The SubNetwork class stacks multiple HebbianLayers to create a more complex subnetwork.
class SubNetwork(nn.Module):
    def __init__(self, dim, num_layers):
        super(SubNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            scaling = torch.FloatTensor(1).uniform_(0.0001, 0.05).item()
            decay = torch.FloatTensor(1).uniform_(0.0001, 0.01).item()
            self.layers.append(HebbianLayer(dim, scaling, decay))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 7. Neural Memory Network
# The NeuralMemoryNetwork class combines multiple subnetworks and an emotional module to form a memory network that can update its weights based on neurotransmitter signals.    
class NeuralMemoryNetwork(nn.Module):
    def __init__(self, num_subnetworks, num_layers):
        super(NeuralMemoryNetwork, self).__init__()
        self.subnetworks = nn.ModuleList()
        self.hidden_size = INTERNAL_DIM // num_subnetworks
        assert INTERNAL_DIM % num_subnetworks == 0, "INTERNAL_DIM must be divisible by num_subnetworks"
        self.num_subnetworks = num_subnetworks
        self.num_layers = num_layers
        for _ in range(num_subnetworks):
            self.subnetworks.append(SubNetwork(self.hidden_size, num_layers))

        self.output_layer = nn.Linear(INTERNAL_DIM, INTERNAL_DIM)
        self.layer_norm = nn.LayerNorm(INTERNAL_DIM)  # Apply layer normalization

        self.prev_loss = None
        self.activations = None

    def forward(self, x):
        batch_size = x.size(0)  # (batch_size, INTERNAL_DIM)
        x_splits = x.split(self.hidden_size, dim=-1)  # List of (batch_size, hidden_size) tensors
        activations_of_all_subnetworks = []

        for i, subnetwork in enumerate(self.subnetworks):
            activations = subnetwork(x_splits[i])  # (batch_size, hidden_size)
            activations_of_all_subnetworks.append(activations)

        # Combine final activations for the final output
        combined_activations = torch.cat(activations_of_all_subnetworks, dim=-1)  # (batch_size, INTERNAL_DIM)
        
        # Use combined activations to generate the final output
        final_output = self.output_layer(combined_activations)  # (batch_size, INTERNAL_DIM)
        final_output = self.layer_norm(final_output)  # (batch_size, INTERNAL_DIM)
        self.activations = final_output
        return final_output


# 9. textual Feature Adaptor
# The TextualFeatureAdaptor class adapts text input embedding into the internal dimensionality.
class TextualFeatureAdaptor(nn.Module):
    def __init__(self, input_size, output_size):
        super(TextualFeatureAdaptor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc(x))
    
# 10. Attention Module
# The AttentionModule class applies an attention mechanism to combine multilpe sources of embeddings into 1
class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query_input, key_value_inputs):
        # Prepare for attention mechanism
        key_value_tensor = torch.stack(key_value_inputs)  # Shape: (num_sources, batch_size, internal_dim)
        # Query input
        query = query_input.unsqueeze(0)  # Shape: (1, batch_size, internal_dim)
        

        # print(key_value_tensor.shape, query.shape)
        # Apply attention with query input, and key_value_inputs as key and value
        attn_output, _ = self.attention(query, key_value_tensor, key_value_tensor)
        
        # Remove the singleton dimension from the output
        attn_output = attn_output.squeeze(0)  # Shape: (batch_size, internal_dim)

        return attn_output

# 11. Brain
# The Brain class integrates all the components to simulate a neural network inspired by the human brain, combining vision processing,
# numeric feature adaptation, memory networks, and attention mechanisms.
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.fovea_loc_pred = FoveaPosPredictor()
        self.vision_encoder = VisionEncoder()

        # Neural memory networks
        self.neural_memory_networks = nn.ModuleList([
            NeuralMemoryNetwork(
                num_subnetworks=NUM_SUB_NETWORKS,
                num_layers=NUM_LAYERS,
            ) for _ in range(NUM_MEMORY_NETWORKS)
        ])

        # Attention mechanism
        self.attention = AttentionModule(embed_dim=INTERNAL_DIM, num_heads=8)


    def forward(self, imgs, cls_tokens):
        # print("in brain forward")
        # process peripheral vision for all cameras
        peripheral_encodings = []
        for i, img in enumerate(imgs):
            # print(f"in brain forward, imgs[{i}] shape: ", img.shape)
            peripheral_encoding = self.vision_encoder.forward_peripheral(img)
            # print(f"in brain forward, periph encoding shape: ", peripheral_encoding.shape)
            peripheral_encodings.append(peripheral_encoding)
        # print("peripheral embeddings made")
        # Combine encodings of images peripheral
        peripheral_combined = self.attention(peripheral_encodings[0], peripheral_encodings[1:])
        # print("peripheral embeddings combined")
        # store all memory network outputs here
        final_outputs_of_all_memory_networks = []
        
        # write peripheral vision experience to memory
        for nmn in self.neural_memory_networks:
            final_outputs_of_all_memory_networks.append(nmn(peripheral_combined))
        # print("peripheral embeddings written to memory")
        # Process fovea vision using CLS tokens
        for cls in cls_tokens:
            fovea_encodings = []
            for i, img in enumerate(imgs):
                # Predict fovea x and y coordinates
                fovea_coords = self.fovea_loc_pred(peripheral_encodings[i], cls)
                # Process fovea using the predicted coordinates
                fovea_encoding = self.vision_encoder.forward_fovea(img, fovea_coords)
                # print(f"in brain forward, fovea encoding shape: ", fovea_encoding.shape)
                fovea_encodings.append(fovea_encoding)

            # Combine fovea encodings using the attention module
            fovea_combined = self.attention(fovea_encodings[0], fovea_encodings[1:])
            # print(f"in brain forward, fovea combined shape: ", fovea_combined.shape)
            # Pass the combined fovea encoding into neural memory networks
            for nmn in self.neural_memory_networks:
                nmn_output = nmn(fovea_combined)
                final_outputs_of_all_memory_networks.append(nmn_output)
            # print("fovea embeddings written to memory")

        # Stack the outputs along a new sequence dimension
        final_outputs_of_all_memory_networks = torch.stack(final_outputs_of_all_memory_networks, dim=1)  # (batch, seqlen, dim)
        # print("returning all memory outputs")

        return final_outputs_of_all_memory_networks