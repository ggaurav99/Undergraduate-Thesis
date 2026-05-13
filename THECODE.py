import os
os.environ["AUTO_DIR"] = "/Users/gaurav/auto-07p"

# Then your existing imports
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pyrates.frontend import OperatorTemplate, NodeTemplate, CircuitTemplate
from pycobi import ODESystem
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pyrates.frontend import OperatorTemplate, NodeTemplate, CircuitTemplate
from pycobi import ODESystem
import numpy as np


op = OperatorTemplate(
    name="epi_behav_op",
    equations=[

        # =====================
        # Epidemiological model
        # =====================

        "d/dt * S_v = -beta_n * S_v * (I_v + T_v) + omega * R_v - nu * x_V * S_v",
        "d/dt * V_v = -(1 - eps_V) * beta_n * V_v * (I_v + T_v) + nu * x_V * S_v",
        "d/dt * I_v = beta_n * (S_v + (1 - eps_V) * V_v) * (I_v + T_v) - (gamma_n + tau * x_A) * I_v",
        "d/dt * T_v = tau * x_A * I_v - gamma_T * T_v",
        "d/dt * R_v = gamma_n * I_v + gamma_T * T_v - omega * R_v",

        # =====================
        # Derived quantities
        # =====================

        "lambda_v = (1 - eps_V) * beta_n * (I_v + T_v)",
        "lambda_a = beta_n * (S_v + (1 - eps_V) * V_v) * (I_v + T_v)",
        "pi_S_A = (1 - eps_A) * pi_S",

        # =====================
        # Prospect weighting
        # =====================

        "w_lambda_v = exp(-a * (-log(lambda_v + 1e-4))**b)",
        "w_lambda_a = exp(-a * (-log(lambda_a + 1e-4))**b)",
        "w_lambda_piS_A = exp(-a * (-log(lambda_a * pi_S_A + 1e-4))**b)",
        "w_lambda_piS_A_piD = exp(-a * (-log(lambda_a * pi_S_A * pi_D + 1e-4))**b)",
        "w_rho_R = exp(-a * (-log(rho_R + 1e-4))**b)",
        "w_tau_v = exp(-a * (-log(tau_v + 1e-4))**b)",

        # =====================
        # Discounting
        # =====================

        "delta_tau_v = 1 / (1 + k * tau_v)",
        "delta_tau_A = 1 / (1 + k * tau_A)",
        "delta_tau_S = 1 / (1 + k * tau_S)",
        "delta_tau_D = 1 / (1 + k * tau_D)",
        "delta_tau_R = 1 / (1 + k * tau_R)",

        # =====================
        # Strategy closure
        # =====================

        "x_N = 1 - x_V - x_A",

        # =====================
        # Replicator dynamics (payoffs fully inlined)
        # =====================

        "d/dt * x_V = 0.1* x_V * ("
        "  (-(delta_tau_v * c_V + w_lambda_v * delta_tau_S * (pi_S * L_S + pi_S * pi_D * L_D)))"
        "  - (x_V * (-(delta_tau_v * c_V + w_lambda_v * delta_tau_S * (pi_S * L_S + pi_S * pi_D * L_D)))"
        "     + x_A * (-(w_lambda_a * delta_tau_A * c_A + w_lambda_a * pi_S_A * delta_tau_S * L_S + w_lambda_a * pi_S_A * pi_D * delta_tau_D * L_D + w_rho_R * delta_tau_R * Omega_R))"
        "     + (1 - x_V - x_A) * (-c_N -(w_lambda_a * delta_tau_S * (pi_S * L_S + pi_S * pi_D * L_D))))"
        ")  + 1e-6 * (1/3 - x_V)",

        # x_A equation

        "d/dt * x_A =  0.1 * x_A * ("
        "  (-(w_lambda_a * delta_tau_A * c_A + w_lambda_a * pi_S_A * delta_tau_S * L_S + w_lambda_a *pi_S_A * pi_D * delta_tau_D * L_D + w_rho_R * delta_tau_R * Omega_R))"
        "  - (x_V * (-(delta_tau_v * c_V + w_lambda_v * delta_tau_S * (pi_S * L_S + pi_S * pi_D * L_D)))"
        "     + x_A * (-(w_lambda_a * delta_tau_A * c_A + w_lambda_a * pi_S_A * delta_tau_S * L_S + w_lambda_a * pi_S_A * pi_D * delta_tau_D * L_D + w_rho_R * delta_tau_R * Omega_R))"
        "     + (1 - x_V - x_A) * (-(w_lambda_a * delta_tau_S * (c_N + pi_S * L_S + pi_S * pi_D * L_D))))"
        " ) + 1e-6 * (1/3 - x_A) ",

    ],

    variables={

        # Dynamical Variables
        "S_v": "variable",
        "V_v": "variable",
        "I_v": "variable",
        "T_v": "variable",
        "R_v": "variable",
        "x_V": "variable",
        "x_A": "variable",

        # Intermediates
        "lambda_v": 0.0,
        "lambda_a": 0.0,
        "pi_S_A": 0.0,

        "w_lambda_v": 0.0,
        "w_lambda_a": 0.0,
        "w_lambda_piS_A": 0.0,
        "w_lambda_piS_A_piD": 0.0,
        "w_rho_R": 0.0,
        "w_tau_v": 0.0,

        "delta_tau_v": 0.0,
        "delta_tau_A": 0.0,
        "delta_tau_S": 0.0,
        "delta_tau_D": 0.0,
        "delta_tau_R": 0.0,

        "x_N": 0.0,
        

        # Parameters
    "beta_n": 0.55,      
    "gamma_n": 0.08,
    "gamma_T": 0.15,
    "tau": 0.15,

    # --- Probabilities ---
    "pi_S": 0.3,
    "pi_D": 0.05,       

    # --- Vaccine ---
    "eps_V": 0.80,      
    "c_V": 0.8,

    # --- Antibiotic ---
    "c_A": 1.0,         
    "eps_A": 0.5,       

    # --- Losses (REDUCED from before) ---
    "L_S": 60.0,
    "L_D": 300.0,

    # --- Behaviour ---
    "a": 1.0,
    "b": 0.70,
    "k": 0.1,

    # --- Delays ---
    "tau_v": 0.1,
    "tau_A": 0.3,
    "tau_S": 1.0,
    "tau_D": 2.0,
    "tau_R": 5.0,

    # --- Resistance ---
    "rho_R": 0.6,       
    "Omega_R": 6.0,    

    # --- Behaviour dynamics ---
    "omega": 0.008,
    "nu": 0.05,

    # --- Baseline penalty for N ---
    "c_N": 1.5

    }
)


node = NodeTemplate(name="node", operators=[op])
circuit = CircuitTemplate(name="epi_behav_circuit", nodes={"node": node})


sys = ODESystem.from_template(
    circuit,
    auto_dir="/Users/gaurav/auto-07p",
    working_dir="results",
    init_cont=False
)


# ======================
# Time Integration
# ======================

t_sols, t_cont = sys.run(
    c="ivp",
    name="time",
    DS=0.01,
    NPR = 100,
    NMX=200000,
    DSMAX=0.01,

    U={
        1: 0.88,  # S
        2: 0.005,  # V
        3: 0.08,  # I
        4: 0.01,  # T
        5: 0.025,  # R
        6: 0.05,  # x_V
        7: 0.20,  # x_A
    },

    UZR={14: 2000.0},   # create UZ1 at 
    STOP={"UZ1"}       # stop exactly there
)


# ======================
# Population conservation check
# ======================

pop_sum = (
    t_sols["node/epi_behav_op/S_v"]
    + t_sols["node/epi_behav_op/V_v"]
    + t_sols["node/epi_behav_op/I_v"]
    + t_sols["node/epi_behav_op/T_v"]
    + t_sols["node/epi_behav_op/R_v"]
)

print("Minimum population sum:", pop_sum.min())
print("Maximum population sum:", pop_sum.max())

# ======================
# Lambda max check
# ======================
S = t_sols["node/epi_behav_op/S_v"]
V = t_sols["node/epi_behav_op/V_v"]
I = t_sols["node/epi_behav_op/I_v"]
T = t_sols["node/epi_behav_op/T_v"]

lambda_v_series = (1 - op.variables["eps_V"]) * op.variables["beta_n"] * (I + T)
lambda_a_series = op.variables["beta_n"] * (S + (1 - op.variables["eps_V"]) * V) * (I + T)

print("Max lambda_v:", lambda_v_series.max())
print("Max lambda_a:", lambda_a_series.max())




# # ======================
# # Population sum every 10 steps
# # ======================

# with open("pop_sum_check.txt", "w") as f:
#     f.write("step\t t\t\t pop_sum\n")
#     f.write("-" * 40 + "\n")
#     for i in range(0, len(pop_sum), 10):
#         t_val = float(t_sols["t"].iloc[i])
#         p_val = float(pop_sum.iloc[i])
#         f.write(f"{i}\t {t_val:.6f}\t {p_val:.10f}\n")

# print("Population sum check saved to pop_sum_check.txt")

# ======================
# Save time series
# ======================

variables = [
    "node/epi_behav_op/S_v",
    "node/epi_behav_op/V_v",
    "node/epi_behav_op/I_v",
    "node/epi_behav_op/T_v",
    "node/epi_behav_op/R_v",
    "node/epi_behav_op/x_V",
    "node/epi_behav_op/x_A"
]

# data = np.column_stack(
#     [t_sols["t"]] + [t_sols[var] for var in variables]
# )

# header = "t  S  V  I  T  R  x_V  x_A"

# np.savetxt(
#     "time_series.txt",
#     data,
#     header=header,
#     fmt="%.8e"
# )

# ======================
# Plot time series
# ======================

# ======================
# Plot time series with poster-optimized layout
# ======================

# Set font sizes for poster readability
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 3

# Create SQUARISH figure (8x8 inches instead of 10x8)
fig, axes = plt.subplots(2, 1, figsize=(13, 7))

# Epidemiological compartments
axes[0].plot(t_sols["t"], t_sols["node/epi_behav_op/S_v"], label="S", linewidth=3)
axes[0].plot(t_sols["t"], t_sols["node/epi_behav_op/V_v"], label="V", linewidth=3)
axes[0].plot(t_sols["t"], t_sols["node/epi_behav_op/I_v"], label="I", linewidth=3)
axes[0].plot(t_sols["t"], t_sols["node/epi_behav_op/T_v"], label="T", linewidth=3)
axes[0].plot(t_sols["t"], t_sols["node/epi_behav_op/R_v"], label="R", linewidth=3)
axes[0].set_xlabel("Time (days)", fontsize=15, fontweight='bold')
axes[0].set_ylabel("Population fraction", fontsize=15, fontweight='bold')
axes[0].set_title("Epidemiological Compartments", fontsize=15, fontweight='bold')
axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].tick_params(axis='both', which='major', width=2, length=8)

# Behavioural strategies
axes[1].plot(t_sols["t"], t_sols["node/epi_behav_op/x_V"], label="x_V (Vaccinate)", linewidth=3)
axes[1].plot(t_sols["t"], t_sols["node/epi_behav_op/x_A"], label="x_A (Antibiotic)", linewidth=3)
axes[1].set_xlabel("Time (days)", fontsize=15, fontweight='bold')
axes[1].set_ylabel("Strategy fraction", fontsize=15, fontweight='bold')
axes[1].set_title("Behavioural Strategy Dynamics", fontsize=15, fontweight='bold')
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].tick_params(axis='both', which='major', width=2, length=8)

plt.tight_layout()
plt.savefig("time_series.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print("Done. Poster-ready time series saved to time_series.png")


# sols_k, cont_k = sys.run(
#     origin=t_cont,   # ← MUST be present
#     starting_point="EP",
#     name="bif_a",

#     ICP="node/epi_behav_op/k",
#     IPS=1,
#     ILP=1,
#     ISP=2,
#     ISW=-1,

#     RL0=0.01,
#     RL1=5.0,
#     DS=0.001,
#     NMX=10000,
#     NPR = 50,


# )

# fig, ax = plt.subplots()

# sys.plot_continuation(
#     "node/epi_behav_op/k",
#     "node/epi_behav_op/x_V",
#     cont="bif_a",
#     ax=ax
# )

# ax.set_title("Bifurcation Diagram")

# fig.savefig(
#     "bifurcation.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.close(fig)



# import matplotlib.pyplot as plt
# import numpy as np

# data = sols_k  # pandas DataFrame

# k_col     = "node/epi_behav_op/k"
# xv_col    = "node/epi_behav_op/x_V"
# xa_col    = "node/epi_behav_op/x_A"
# type_col  = "bifurcation"

# # bifurcation points
# bp = data[data[type_col] == "BP"]
# lp = data[data[type_col] == "LP"]
# hb = data[data[type_col] == "HB"]

# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # --- x_V panel ---
# axes[0].scatter(data[k_col], data[xv_col], s=4, color='steelblue', alpha=0.4, label='equilibria')
# axes[0].scatter(bp[k_col], bp[xv_col], s=20, color='blue',   label='BP', zorder=5)
# axes[0].scatter(lp[k_col], lp[xv_col], s=20, color='orange', label='LP', zorder=5)
# axes[0].scatter(hb[k_col], hb[xv_col], s=20, color='green',  label='HB', zorder=5)
# # axes[0].axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='baseline $k=0.5$')
# axes[0].set_ylabel("$x_V$", fontsize=12)
# axes[0].set_title("Bifurcation diagram", fontsize=13)
# axes[0].legend(fontsize=9)
# axes[0].set_ylim(-0.05, 1.05)

# # --- x_A panel ---
# axes[1].scatter(data[k_col], data[xa_col], s=4, color='darkorange', alpha=0.4, label='equilibria')
# axes[1].scatter(bp[k_col], bp[xa_col], s=20, color='blue',   label='BP', zorder=5)
# axes[1].scatter(lp[k_col], lp[xa_col], s=20, color='orange', label='LP', zorder=5)
# axes[1].scatter(hb[k_col], hb[xa_col], s=20, color='green',  label='HB', zorder=5)
# # axes[1].axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='baseline $k=0.5$')
# axes[1].set_xlabel("$k$", fontsize=12)
# axes[1].set_ylabel("$x_A$", fontsize=12)
# axes[1].legend(fontsize=9)
# axes[1].set_ylim(-0.05, 1.05)

# plt.tight_layout()
# fig.savefig("bifurcation_xV_xA.png", dpi=300, bbox_inches="tight")
# plt.close(fig)
# print("Saved bifurcation_xV_xA.png")