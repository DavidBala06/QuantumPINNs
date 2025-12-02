**An Inverse Physics-Informed Neural Network that "discovers" hidden physical properties of materials by observing electron behavior.**
<img width="1536" height="754" alt="QuantumPinns" src="https://github.com/user-attachments/assets/20a521b6-afb8-4555-a67f-b0c5b06d08a9" />


## 📖 Overview
In traditional engineering and physics, we usually solve **Forward Problems**: *Given the material properties, how will the particle behave?*

This project solves the harder, more valuable **Inverse Problem**: *Given the observed particle behavior, what are the material properties?*

Using **PyTorch** and **Automatic Differentiation**, I built a model that:
1.  Takes raw observations of a quantum wave function psi.
2.  Embeds the **Schrödinger Equation** directly into the Neural Network's loss function.
3.  Backpropagates through the physics equations to learn the unknown potential parameter k (stiffness) of the material.

## 🚀 Key Features
* **Physics-Informed Loss:** The model is trained not just on data, but on the physical laws of the universe (Loss = Loss_{Data} + Loss_{Schrödinger}).
* **Inverse Parameter Estimation:** The network treats the physical constant $k$ as a trainable parameter, discovering its true value from a random initialization.
* **Scientific Machine Learning (SciML):** Demonstrates the use of AI for system identification and digital twin modeling.

## 🧠 The Physics (Math Backend)
The project models an electron in a **Quantum Harmonic Oscillator**. The behavior is governed by the Time-Dependent Schrödinger Equation:

<img width="457" height="132" alt="Captură de ecran 2025-12-02 184133" src="https://github.com/user-attachments/assets/c69559d3-b71b-4ee1-a152-d1831a9638c7" />


Where the potential $V(x)$ describes the material:
<img width="103" height="51" alt="Captură de ecran 2025-12-02 184339" src="https://github.com/user-attachments/assets/561908b2-b67f-4dd2-bbe5-ec6a7878b6fa" />


**The Challenge:** The AI is **not told** the value of $k$. It must deduce that $k=1.0$ solely by minimizing the physical residuals of the wave function derivatives.

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/DavidBala06/quantum-pinn-material-discovery.git](https://github.com/DavidBala06/quantum-pinn-material-discovery.git)
    cd quantum-pinn-material-discovery
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the discovery engine**
    ```bash
    python main.py
    ```
    *The training will begin, and you will see the estimated `k` value converge in real-time in the console.*

## 📊 Results
As seen in the dashboard above:
* **Ground Truth:** k = 1.0
* **Initial Guess:** k = 0.1
* **Final Discovery:** k \approx 0.96
* **Convergence:** The model successfully identified the material properties with **>95% accuracy** purely from observational data, without being given the answer key.

## 📂 Project Structure
```text
├── main.py                     # The core PINN logic and training loop
├── requirements.txt            # Python dependencies
├── pinn_results_dashboard.png  # Visualization of the results
├── README.md                   # Project documentation
└── .gitignore                  # Git configuration
