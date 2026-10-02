import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import pymatviz as pmv
from ase.io import read
from ase.db import connect
from mace.calculators import MACECalculator

def generate_figures():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Trouve automatiquement le fichier modèle généré
    model_files = [f for f in os.listdir(".") if f.endswith(".model")]
    if not model_files:
        print("Erreur : Aucun modèle MACE trouvé dans le dossier.")
        return
    model_path = model_files[0]
    
    device = config.get("mace_mlip", {}).get("device", "cpu")
    calc = MACECalculator(model_paths=model_path, device=device)
    db = connect(config["system"]["db_path"])
    os.makedirs("figures", exist_ok=True)
    
    val_file = config["system"]["export_val_xyz"]
    val_atoms = read(val_file, index=":")
    
    print(f"Génération du graphique d'énergie totale ({len(val_atoms)} structures)...")
    dft_tot, mace_tot = [], []
    for atoms in val_atoms:
        dft_tot.append(atoms.info.get("energy", atoms.get_potential_energy()))
        atoms_pred = atoms.copy()
        atoms_pred.calc = calc
        mace_tot.append(atoms_pred.get_potential_energy())
        
    df_tot = pd.DataFrame({"DFT": dft_tot, "MACE": mace_tot})
    mae_tot = (df_tot["DFT"] - df_tot["MACE"]).abs().mean()
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    pmv.density_scatter(df_tot["DFT"], df_tot["MACE"], ax=ax)
    min_tot, max_tot = min(df_tot.min()), max(df_tot.max())
    ax.plot([min_tot, max_tot], [min_tot, max_tot], 'k--', alpha=0.7, label="Parité parfaite")
    
    ax.set_xlabel("Énergie Totale DFT (eV)", fontweight="bold")
    ax.set_ylabel("Énergie Totale MACE (eV)", fontweight="bold")
    ax.set_title(f"MACE Total Energy (MAE = {mae_tot:.3f} eV)", fontweight="bold")
    ax.legend()
    plt.savefig("figures/mace_total_energy_parity.png")
    plt.close()
    
    print("Graphique d'énergie totale sauvegardé dans figures/mace_total_energy_parity.png")

if __name__ == "__main__":
    generate_figures()