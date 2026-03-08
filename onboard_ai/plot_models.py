import matplotlib.pyplot as plt

# Data for YOLO26
yolo_models = ['YOLO26n', 'YOLO26s', 'YOLO26m', 'YOLO26l', 'YOLO26x']
yolo_params = [2.4, 9.5, 20.4, 24.8, 55.7]
yolo_gflops = [5.4, 20.7, 68.2, 86.4, 193.9]
yolo_ap_50 = [55.8, 64.3, 69.7, 71.1, 74.0]  # AP50 values for YOLO26
yolo_ap_5095 = [40.9, 48.6, 53.1, 55.0, 57.5]  # AP50:95 values for YOLO26
yolo_size = ['N', 'S', 'M', 'L', 'XL']

# Data for DEIMv2
deim_models = ['Atto', 'Femto', 'Pico', 'N', 'S', 'M', 'L', 'X']
deim_params = [0.5, 1.0, 1.5, 3.6, 9.7, 18.1, 32.2, 50.3]
deim_gflops = [0.8, 1.7, 5.2, 6.8, 25.6, 52.2, 96.7, 151.6]
deim_ap_50 = [None, None, None, None, 68.3, 70.2, 73.4, 75.4]  # AP50 values for DEIMv2
deim_ap_5095 = [23.8, 31.0, 38.5, 43.0, 50.9, 53.0, 56.0, 57.8]  # AP50:95 values for DEIMv2
deim_size = ['Atto', 'Femto', 'Pico', 'N', 'S', 'M', 'L', 'X']

# Select which AP to show: 'AP50', 'AP50:95', or 'both'
show_ap = 'AP50'  # Change to 'AP50:95' or 'both' to show other options

# Choose the appropriate values to plot based on the selected 'show_ap'
if show_ap == 'AP50':
    yolo_ap = yolo_ap_50
    deim_ap = deim_ap_50
elif show_ap == 'AP50:95':
    yolo_ap = yolo_ap_5095
    deim_ap = deim_ap_5095
else:  # show 'both'
    yolo_ap = yolo_ap_50
    deim_ap = deim_ap_50

# Remove None values from DEIMv2
deim_params_cleaned = [deim_params[i] for i in range(len(deim_params)) if deim_ap[i] is not None]
deim_ap_cleaned = [deim_ap[i] for i in range(len(deim_ap)) if deim_ap[i] is not None]
deim_size_cleaned = [deim_size[i] for i in range(len(deim_size)) if deim_ap[i] is not None]
deim_gflops_cleaned = [deim_gflops[i] for i in range(len(deim_size)) if deim_ap[i] is not None]

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Parameters vs mAP
# YOLO26 data (line plot)
ax1.plot(yolo_params, yolo_ap, color='tab:blue', label="YOLO26")
for i in range(len(yolo_params)):
    ax1.text(yolo_params[i], yolo_ap[i], yolo_size[i], color='tab:blue', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))  # Reduced padding

# DEIMv2 data (line plot) - Only plot models with non-None AP50
ax1.plot(deim_params_cleaned, deim_ap_cleaned, color='tab:orange', label="DEIMv2")
for i in range(len(deim_params_cleaned)):
    ax1.text(deim_params_cleaned[i], deim_ap_cleaned[i], deim_size_cleaned[i], color='tab:orange', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))  # Reduced padding

ax1.set_xlabel('Number of Parameters (M)', fontsize=16)
ax1.set_ylabel(f'{show_ap}', fontsize=16)  # Title based on the selected AP
ax1.set_title(f'Parameters vs {show_ap}', fontsize=16)
ax1.legend()
ax1.grid(True)  # Add grid to the plot

# Plot 2: GFLOPs vs mAP
# YOLO26 data (line plot)
yolo_gflops = [5.4, 20.7, 68.2, 86.4, 193.9]
ax2.plot(yolo_gflops, yolo_ap, color='tab:blue', label="YOLO26")
for i in range(len(yolo_gflops)):
    ax2.text(yolo_gflops[i], yolo_ap[i], yolo_size[i], color='tab:blue', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))  # Reduced padding

# DEIMv2 data (line plot) - Only plot models with non-None AP50
ax2.plot(deim_gflops_cleaned, deim_ap_cleaned, color='tab:orange', label="DEIMv2")
for i in range(len(deim_gflops_cleaned)):
    ax2.text(deim_gflops_cleaned[i], deim_ap_cleaned[i], deim_size_cleaned[i], color='tab:orange', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))  # Reduced padding

ax2.set_xlabel('GFLOPs', fontsize=16)
# ax2.set_ylabel(f'{show_ap}', fontsize=16)  # Title based on the selected AP
ax2.set_title(f'GFLOPs vs {show_ap}', fontsize=16)
ax2.legend()
ax2.grid(True)  # Add grid to the plot

plt.tight_layout()
plt.show()