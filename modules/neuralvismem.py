import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the input and output dimensions for internal modules
INTERNAL_DIM = 768
NUM_SUB_NETWORKS = 8
NUM_LAYERS = 5
NUM_MEMORY_NETWORKS = 16
NUM_NEURO_TRANSMITTERS = 4
EMOTIONAL_DIM = 256
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
    def __init__(self, dim, scaling, decay):
        super(HebbianLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.recurrent_weight = nn.Parameter(torch.randn(dim, dim))
        self.alpha = scaling  # Scaling parameter for Hebbian learning
        self.decay = decay  # Decay parameter
        self.prev_state = None
        self.prev_input = None
        self.layer_norm = nn.LayerNorm(dim)  # Layer normalization for the activations
        self.max_magnitude = 3

        # Neurotransmitter signals
        self.dopamine_signal = None
        self.serotonin_signal = None
        self.gaba_signal = None
        self.glutamate_signal = None

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        
        if self.prev_state is None:
            self.prev_state = torch.zeros(batch_size, dim, device=x.device)
        
        if self.dopamine_signal is None:
            self.dopamine_signal = torch.zeros(batch_size, device=x.device)
            self.serotonin_signal = torch.zeros(batch_size, device=x.device)
            self.gaba_signal = torch.zeros(batch_size, device=x.device)
            self.glutamate_signal = torch.zeros(batch_size, device=x.device)

        # Recurrent weight multiplication
        combined_input = x + torch.matmul(self.prev_state, self.recurrent_weight.t())
        
        # Apply input weights
        activations = torch.matmul(combined_input, self.weight.t())
        activations = self.layer_norm(activations)

        # Apply GABA and glutamate modulation
        threshold = torch.mean(activations, dim=1, keepdim=True)
        threshold += self.gaba_signal.unsqueeze(1)
        threshold -= self.glutamate_signal.unsqueeze(1)
        activations = torch.relu(activations - threshold)
        activations = self.layer_norm(activations)

        # Store current state for Hebbian update and next step
        self.prev_input = x.detach().clone()  # Detach to avoid gradient computation
        self.prev_state = activations.detach().clone()

        return activations

    def hebbian_update(self):
        with torch.no_grad():
            hebbian_term_input = torch.matmul(self.prev_state.t(), self.prev_input)
            batch_size = self.dopamine_signal.size(0)
            dopamine_signal = self.dopamine_signal.view(batch_size, 1, 1)

            weight_update = self.alpha * dopamine_signal * hebbian_term_input.unsqueeze(0) / batch_size
            self.weight.data = self.weight.data + weight_update.sum(dim=0)

            hebbian_term_recurrent = torch.matmul(self.prev_state.t(), self.prev_state)
            recurrent_weight_update = self.alpha * dopamine_signal * hebbian_term_recurrent.unsqueeze(0) / batch_size
            self.recurrent_weight.data = self.recurrent_weight.data + recurrent_weight_update.sum(dim=0)

            # Apply decay
            self.weight.data = self.weight.data - self.decay * self.weight.data
            self.recurrent_weight.data = self.recurrent_weight.data - self.decay * self.recurrent_weight.data

            # Normalize weights
            self.weight.data = nn.functional.normalize(self.weight.data, p=2, dim=1) * self.max_magnitude
            self.recurrent_weight.data = nn.functional.normalize(self.recurrent_weight.data, p=2, dim=1) * self.max_magnitude

    def update_neuro_modulators(self, dopamine, serotonin, gaba, glutamate):
        self.dopamine_signal = dopamine + (dopamine - self.dopamine_signal) * (1 - serotonin)
        self.serotonin_signal = serotonin
        self.gaba_signal = gaba
        self.glutamate_signal = glutamate

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

    def update_neuro_modulators(self, neurotransmitter_signals):
        # neurotransmitter_signals: (batch_size, num_layers, 4)
        batch_size = neurotransmitter_signals.size(0)
        num_layers = neurotransmitter_signals.size(1)
        
        for i, layer in enumerate(self.layers):
            # Extract neurotransmitter signals for the current layer across the batch
            dopamine_signal = neurotransmitter_signals[:, i, 0]  # (batch_size,)
            serotonin_signal = neurotransmitter_signals[:, i, 1]  # (batch_size,)
            gaba_signal = neurotransmitter_signals[:, i, 2]  # (batch_size,)
            glutamate_signal = neurotransmitter_signals[:, i, 3]  # (batch_size,)
            
            # Update neurotransmitters for the entire batch at once
            layer.update_neuro_modulators(
                dopamine_signal,
                serotonin_signal,
                gaba_signal,
                glutamate_signal
            )

# 6. EmotionalModule
# The EmotionalModule class generates dopamine, serotonin, GABA, etc. signals based on the sensory input and the activations from the networks.
class EmotionalModule(nn.Module):
    def __init__(self, input_size, num_subnetworks, num_layers):
        super(EmotionalModule, self).__init__()
        self.fc1 = nn.Linear(input_size, EMOTIONAL_DIM)
        self.fc2 = nn.Linear(EMOTIONAL_DIM, num_subnetworks * num_layers * NUM_NEURO_TRANSMITTERS)
        self.num_layers = num_layers
        self.num_neurotransmitters = NUM_NEURO_TRANSMITTERS

    def forward(self, combined_sensory_encoding, activations):
        # Ensure activations are detached to prevent gradients from flowing back
        activations = activations.detach()
        batch_size = combined_sensory_encoding.size(0)

        # Concatenate along the feature dimension
        x = torch.cat([combined_sensory_encoding, activations], dim=-1)  # Shape: (batch, 2 * dim)
        x = F.relu(self.fc1(x))  # Shape: (batch, EMOTIONAL_DIM)

        # Generate neurotransmitter signals
        neurotransmitter_signals = torch.tanh(self.fc2(x))  # Output range [-1, 1]
        # Shape: (batch, num_subnetworks * num_layers * num_neurotransmitters)

        # Reshape the output to (batch, num_subnetworks, num_layers, num_neurotransmitters)
        neurotransmitter_signals = neurotransmitter_signals.view(
            batch_size, -1, self.num_layers, self.num_neurotransmitters
        )

        return neurotransmitter_signals  # Shape: (batch, num_subnetworks, num_layers, num_neurotransmitters)

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
        self.emotional_module = EmotionalModule(INTERNAL_DIM + INTERNAL_DIM, num_subnetworks, num_layers)
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

    def update_neuro_modulators(self, combined_sensory_encoding):
        # Get activations
        activations = self.activations.detach()  # (batch_size, INTERNAL_DIM)

        # Compute neurotransmitter signals
        neurotransmitter_signals = self.emotional_module(combined_sensory_encoding, activations)  # (batch_size, num_subnetworks, num_layers, NUM_NEURO_TRANSMITTERS)

        # Iterate through each subnetwork and update neurotransmitters
        for i, subnetwork in enumerate(self.subnetworks):
            subnetwork.update_neuro_modulators(neurotransmitter_signals[:, i, :, :])  # (batch_size, num_layers, NUM_NEURO_TRANSMITTERS)


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

        self.peripheral_combined = None
        self.fovea_combined = None

    def forward(self, imgs, cls_tokens):
        print("in brain forward")
        # process peripheral vision for all cameras
        peripheral_encodings = []
        for i, img in enumerate(imgs):
            # print(f"in brain forward, imgs[{i}] shape: ", img.shape)
            peripheral_encoding = self.vision_encoder.forward_peripheral(img)
            # print(f"in brain forward, periph encoding shape: ", peripheral_encoding.shape)
            peripheral_encodings.append(peripheral_encoding)
        print("peripheral embeddings made")
        # Combine encodings of images peripheral
        peripheral_combined = self.attention(peripheral_encodings[0], peripheral_encodings[1:])
        self.peripheral_combined = peripheral_combined.clone().detach()  # save for emotional module in hebbian update
        print("peripheral embeddings combined")
        # store all memory network outputs here
        final_outputs_of_all_memory_networks = []
        
        # write peripheral vision experience to memory
        for nmn in self.neural_memory_networks:
            final_outputs_of_all_memory_networks.append(nmn(peripheral_combined))
            nmn.update_neuro_modulators(self.peripheral_combined)
        print("peripheral embeddings written to memory")
        # Process fovea vision using CLS tokens
        for cls in cls_tokens:
            fovea_encodings = []
            for i, img in enumerate(imgs):
                # Predict fovea x and y coordinates
                fovea_coords = self.fovea_loc_pred(peripheral_encodings[i], cls)
                # Process fovea using the predicted coordinates
                fovea_encoding = self.vision_encoder.forward_fovea(img, fovea_coords)
                print(f"in brain forward, fovea encoding shape: ", fovea_encoding.shape)
                fovea_encodings.append(fovea_encoding)

            # Combine fovea encodings using the attention module
            fovea_combined = self.attention(fovea_encodings[0], fovea_encodings[1:])
            self.fovea_combined = fovea_combined.clone().detach()  # Save for use in neuro modulator updates
            print(f"in brain forward, fovea combined shape: ", fovea_combined.shape)
            # Pass the combined fovea encoding into neural memory networks
            for nmn in self.neural_memory_networks:
                nmn_output = nmn(fovea_combined)
                final_outputs_of_all_memory_networks.append(nmn_output)
                nmn.update_neuro_modulators(self.fovea_combined)
            print("fovea embeddings written to memory")

        # Stack the outputs along a new sequence dimension
        final_outputs_of_all_memory_networks = torch.stack(final_outputs_of_all_memory_networks, dim=1)  # (batch, seqlen, dim)
        print("returning all memory outputs")

        return final_outputs_of_all_memory_networks