import torch
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import spikegen

# 1. Initialize a LIF Neuron
# beta (decay rate) = 0.8
# threshold = 1.0
lif_neuron = snn.Leaky(beta=0.8, threshold=1.0)

# 2. Create some random input data (e.g., pixel intensities)
# 10 time steps, 1 batch, 1 input neuron
torch.manual_seed(0)
input_data = torch.rand(10) 
print(f"Raw Input Values (e.g., Pixel brightness): \n{input_data}\n")

# 3. Spike Encoding (Rate Coding)
# Convert continuous inputs into discrete spikes (1s and 0s)
spike_data = spikegen.rate(input_data, num_steps=10)
print(f"Encoded Spike Train (0s and 1s): \n{spike_data}\n")

# 4. Simulate the Neuron over time
mem = lif_neuron.init_leaky() # Initialize membrane potential to 0

mem_record = []
spike_record = []

print("--- Starting Simulation ---")
for step in range(10):
    # Pass the current time step's spike to the neuron
    current_input = spike_data[step]
    
    # The neuron updates its membrane potential and checks if it fires
    spk, mem = lif_neuron(current_input, mem)
    
    mem_record.append(mem.item())
    spike_record.append(spk.item())
    
    print(f"Step {step+1}: Input={current_input.item():.0f}, Membrane Potential={mem.item():.2f}, Output Spike={spk.item():.0f}")

print("--- Simulation Ended ---")
