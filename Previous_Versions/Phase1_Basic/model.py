import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SpikingQNetwork(nn.Module):
    def __init__(self, num_actions=3, beta=0.9, threshold=1.0):
        """
        A Spiking Convolutional Neural Network (SCNN) for playing DOOM.
        Uses Surrogate Gradients for backpropagation.
        """
        super().__init__()
        
        # The Trick! Using Fast Sigmoid Surrogate Gradient for training
        spike_grad = surrogate.fast_sigmoid()
        
        # We expect an input image of size (1 Channel, 64x64 Pixels)
        
        # Convolutional Layer 1 (Finds basic edges and shapes)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2) # Output size: 16x30x30
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        # Convolutional Layer 2 (Finds complex patterns like monsters)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2) # Output size: 32x13x13
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        # Dense Layer 1 (Connects all features together)
        # Flattened size: 32 channels * 13 width * 13 height = 5408
        self.fc1 = nn.Linear(5408, 128)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        # Output Layer (Maps to the 3 DOOM actions: Left, Right, Shoot)
        self.fc2 = nn.Linear(128, num_actions)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(self, x, num_steps=10):
        """
        Forward pass of the SNN. SNNs run over 'num_steps' time steps.
        :param x: Input image tensor of shape (Batch, 1, 64, 64)
        :param num_steps: Time steps to simulate
        :return: Recorded spikes and membrane potentials over time
        """
        # Initialize the "water level" (membrane potential) of all buckets (neurons) to 0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # To keep track of what the output layer does over time
        spk4_rec = []
        mem4_rec = []

        # Run the simulation over time steps
        for step in range(num_steps):
            # Pass image through Conv1 -> LIF1
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Pass spikes to Conv2 -> LIF2
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Flatten and pass to Dense Layer 1 -> LIF3
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Pass to Output Layer -> LIF4
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            # Record the final output for this time step
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        # Return the stacked results: shape will be (num_steps, Batch, num_actions)
        return torch.stack(spk4_rec), torch.stack(mem4_rec)

if __name__ == "__main__":
    # Test the model with a fake random DOOM frame
    print("Testing SpikingQNetwork...")
    
    # Create a fake input image (Batch=1, Channel=1, 64x64)
    fake_frame = torch.rand(1, 1, 64, 64) 
    
    model = SpikingQNetwork()
    
    # Run the model
    out_spikes, out_mem = model(fake_frame, num_steps=10)
    
    print(f"Output Spikes shape: {out_spikes.shape}") # (10, 1, 3)
    print(f"Output Membrane shape: {out_mem.shape}") # (10, 1, 3)
    
    # Sum the spikes over the 10 time steps to see which action got the most spikes
    action_votes = out_spikes.sum(dim=0)
    print(f"Action Votes (Left, Right, Shoot): {action_votes}")
    print(f"Chosen Action: {torch.argmax(action_votes).item()}")
    
    print("Model test passed successfully!")
