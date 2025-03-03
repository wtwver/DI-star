import re
import matplotlib.pyplot as plt

# List of loss types to extract
loss_types = [
    'action_type_loss',
    'delay_loss',
    'queued_loss',
    'selected_units_loss',
    'selected_units_loss_norm',
    'selected_units_end_flag_loss',
    'target_unit_loss',
    'target_location_loss',
    'total_loss'
]

# Read the log file
with open('default_logger.txt', 'r') as f:
    log_data = f.read()

# Dictionary to store data for each loss type
loss_data = {loss_type: {'iterations': [], 'values': []} for loss_type in loss_types}

# Extract data for each loss type
for loss_type in loss_types:
    pattern = rf'=== Training Iteration (\d+) Result ===.*?\| {loss_type}\s+\|\s+(\d+\.\d+)'
    matches = re.findall(pattern, log_data, re.DOTALL)
    if matches:
        iterations, values = zip(*[(int(m[0]), float(m[1])) for m in matches])
        loss_data[loss_type]['iterations'] = list(iterations)
        loss_data[loss_type]['values'] = list(values)

# Create 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
fig.suptitle('Training Losses Over Iterations', fontsize=16)

# Flatten axes array for easier iteration
axes_flat = axes.flatten()

# Plot each loss type
for idx, loss_type in enumerate(loss_types):
    if loss_data[loss_type]['values']:  # Only plot if we have data
        ax = axes_flat[idx]
        ax.plot(loss_data[loss_type]['iterations'],
                loss_data[loss_type]['values'],
                marker='o',
                linestyle='-',
                label=loss_type)
        ax.set_title(loss_type)
        ax.grid(True)
        ax.legend()

# Set labels
for ax in axes[-1, :]:  # Bottom row
    ax.set_xlabel('Iteration')
for ax in axes[:, 0]:  # Left column
    ax.set_ylabel('Loss Value')

# Hide unused subplots if any
for idx in range(len(loss_types), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Adjust top to fit main title
plt.show()