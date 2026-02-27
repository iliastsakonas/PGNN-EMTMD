from re import L
from physics.base import PhysicsProblem
import torch
import pandas as pd
import numpy as np
import math


class EMTMDProblem(PhysicsProblem):
    """
    EMTMDminverse design problem.

    How to set up your own problem:
    ───────────────────────────────
    1. Define your design parameters in `design_params` (any number of variables) or pass them as variables on initialization.
    2. Set matching bounds in `bounds` — one (min, max) per parameter.
    3. Point `get_data_path()` to your CSV file.
    4. In `load_data()`, map your CSV columns to inputs/targets.
    5. Implement your physics equations in `compute_emtmd_response()`.
    6. Wire it up in `forward_physics()` so the pipeline can call it.
    """

    def __init__(self):
        # Observed / cached data (populated by load_data)
        self._targets = None
        self._input_data = None

        df = pd.read_csv('data/Mem.csv',header=None)
        self.Mem = torch.tensor(df.values, dtype=torch.float32) # Mass matrix of the electromechanical system (109x109): 
                                                                # Inertia terms will be added later (Inductance)
        df = pd.read_csv('data/Cem.csv',header=None)
        self.Cem = torch.tensor(df.values, dtype=torch.float32) # Damping matrix of the electromechanical system (109x190)
                                                                # Damping terms will be added later (Resistance)
        df = pd.read_csv('data/Kem.csv',header=None)
        self.Kem = torch.tensor(df.values, dtype=torch.float32) # Stiffness matrix of the electromechanical system (118x118)
        Force_vec = np.zeros((118,1), dtype = np.complex64)     # Excitation force vector
        Force_vec[3,0] = 1                                      
        self.Force_vec = torch.from_numpy(Force_vec)
        self.w_range = range(500,4451,5)                          # Frequency range

        # Design parameters the NN will predict 
        # List every variable name here. The order must match the
        # columns returned by the NN (predictions[:, 0], [:, 1], …).
        self.design_params = ['L1','L2','L3','L4','L5','L6','L7','L8','L9',
                              'R1','R2','R3','R4','R5','R6','R7','R8','R9'
        ]

        # Bounds: one (min, max) per design parameter 
        # Must be same length as design_params.
        self.bounds = [(1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2), (1e-4, 1e-2),
                       (1, 150), (1, 150), (1, 150), (1, 150), (1, 150), (1, 150), (1, 150), (1, 150), (1, 150)
        ] # 1-9 values: Inductance (L), 10-18 values: Resistance (R) 

    def get_input_output_dims(self):
        input_dim = len(self.w_range)           # Target: Zero amplitude for the displacement of a point for all frequencies in w_range
        output_dim = len(self.design_params)    # auto-sized from your param list
        return input_dim, output_dim

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        # Anagkaio? Den mporw na dhmiourghsw ystera ena tensor me mhdenika values?
        return 'data/emtmd_data.csv'

    def load_data(self, path):
        """
        Load and prepare training data from CSV.
        
        Your CSV should contain:
          - columns for each design parameter   (optional, for dataset-lookup)
          - a column for the target/objective    (what the NN tries to match)
          - columns for any input observations   (if input_dim > 1)

        Returns:
            inputs  – tensor [N, input_dim]
            targets – tensor [N, 1]
        """
        inp = torch.zeros((1,len(self.w_range)))
        obs = torch.zeros((1,1))
        return inp, obs

    def forward_physics(self, inputs, predictions):
        """
        Called by the training loop every epoch.

        Args:
            inputs:      [batch, input_dim]  — the training inputs
            predictions: [batch, output_dim] — current NN-predicted params

        Returns:
            [batch, 1] tensor — computed observable that is compared to targets.

        Access individual predicted parameters with:
            predictions[:, 0]  → first design param (whole batch)
            predictions[:, 1]  → second design param
            ...
        """
        output = self.compute_emtmd_response(predictions)
        return output

    def compute_emtmd_response(self, predictions):
        """
        Core EMTMD physics equations.

        This is where your actual math goes.  Unpack predicted parameters
        by column index, use fixed constants from self.*, and return the
        quantity that the pipeline compares against the target.

        IMPORTANT: use torch operations (not plain Python/numpy) so that
        gradients flow back through the computation for training.

        Args:
            predictions: [batch, output_dim] predicted design parameters
            inputs:      [batch, input_dim]  input data (e.g. frequency)

        Returns:
            [batch, 1] tensor — computed response
        """
        # Set pi for convenience
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18 = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4], predictions[:, 5], predictions[:, 6], predictions[:, 7], predictions[:, 8], predictions[:, 9], predictions[:, 10], predictions[:, 11], predictions[:, 12], predictions[:, 13], predictions[:, 14], predictions[:, 15], predictions[:, 16], predictions[:, 17]
        # Set up the system matrices
        # Clone the M, C, K matrices and F vector, increase their dimensions by 1 
        Cem_new = self.Cem.clone().unsqueeze(0).repeat(len(self.w_range),1,1)  
        Mem_new = self.Mem.clone().unsqueeze(0).repeat(len(self.w_range),1,1)
        Kem_l = self.Kem.clone().unsqueeze(0).repeat(len(self.w_range), 1, 1)
        Force_l = self.Force_vec.clone().unsqueeze(0).repeat(len(self.w_range),1,1)
        # Create the diagonal matrices for L and R values to be added to the Mem and Cem matrices
        L_mat = (torch.diag(torch.cat( (p1, p2, p3, p4, p5, p6, p7, p8, p9)))).repeat(len(self.w_range),1,1)
        R_mat = (torch.diag(torch.cat( (p10, p11, p12, p13, p14, p15, p16, p17, p18)))).repeat(len(self.w_range),1,1)
        Cem_l = torch.cat((torch.cat((Cem_new, torch.zeros(len(self.w_range),109,9) ),2), torch.cat( (torch.zeros(len(self.w_range),9,109),R_mat) ,2 )),1)
        Mem_l = torch.cat((torch.cat((Mem_new, torch.zeros(len(self.w_range),109,9) ),2), torch.cat( (torch.zeros(len(self.w_range),9,109),L_mat) ,2 )),1)
        
        w = torch.tensor(list(self.w_range),dtype = torch.float32)
        s = 1j*2*math.pi*w
        s = s.unsqueeze(-1).unsqueeze(-1)
        # Dynamic Stiffness Matrix
        DSM = (Mem_l*s**2+Cem_l*s+Kem_l)
        # Solve the linear system
        H = torch.linalg.solve(DSM, Force_l)
        # Output: Amplitude values w.r.t w_range for the tip
        output = torch.abs(H[:,99,:]).squeeze(-1)
        return output
    def constraint_loss(self, predictions):
        """Penalty if designs go outside valid bounds."""
        return torch.zeros(1,1)

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        """
        Save EMTMD-specific results and plots.

        Override this to export CSV tables, frequency-response plots, etc.
        The base class default just saves loss curves.
        """
        import os
        #from visualization.plotting import plot_loss_curves, save_emtmd_epoch_results_csv, evaluate_rank
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
