import numpy as np
import matplotlib.pyplot as plt

# Parameters (replace with your actual mean and standard deviation)
mean = 16
stddev = 12

# Generate x values for the plot
x = np.linspace(mean - 4 * stddev, mean + 4 * stddev, 100)

# Calculate the corresponding y values using the normal distribution PDF formula
y = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * stddev**2))

# Plot the PDF
plt.plot(x, y)

# Set labels and title
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Probability Density Function')

# Show the plot
plt.show()
