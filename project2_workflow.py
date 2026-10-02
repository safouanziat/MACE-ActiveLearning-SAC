import os
import numpy as np
import pandas as pd
import pymatviz as pmv
from jobflow import job, Flow, Response
from ase.io import read, write
from ase.db import connect
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from pipeline_engine import SACPipelineEngine

@job
def generate_metal_motifs_job(metal: str, config: dict):
    engine = SACPipelineEngine(config)
    engine.sys_cfg['metal'] = metal
    
    slab = engine.synthesize_pristine_sac()
    motifs = engine.generate_motifs(slab)
    
    motif_files = {}
    os.makedirs("structures_3fold", exist_ok=True)
    for name, atoms_obj in motifs.items():
        fname = f"structures_3fold/structure_{name}.xyz"
        write(fname, atoms_obj)
        motif_files[name] = fname
        
    return {"metal": metal, "motif_files": motif_files}

@job
def run_dft_reference_job(motif_name: str, structure_file: str, config: dict, is_active_learning: bool = False):
    slab = read(structure_file)
    engine = SACPipelineEngine(config)
    engine.sys_cfg['metal'] = motif_name.split('-')[0]
    
    db = connect(config["system"]["db_path"])
    
    # Gestion propre du Smart Resume pour la production et GitHub
    if not is_active_learning:
        existing_data = list(db.select(motif=motif_name, stage="aimd_img0"))
        if len(existing_data) > 0:
            print(f"✅ [Smart Resume] Données AIMD déjà présentes pour {motif_name}. Saut du calcul DFT lourd.")
            return {"motif": motif_name, "qc_passed": True, "images_sampled": 0}
    else:
        print(f"🔄 [Active Learning] Contournement du Smart Resume : Génération de nouvelles données DFT pour {motif_name}.")
    
    images = engine.generate_training_pathway(motif_name, slab)
    
    ts_idx = len(images) // 2
    ts_image = images[ts_idx].copy()
    
    print(f"\n--- [QC] Analyse vibratoire partielle sur {motif_name} ---")
    is_valid_ts = engine.verify_ts_vibrations(ts_image, f"freq_{motif_name}")
    
    if not is_valid_ts:
        print(f"⚠️ Alerte QC : Le MEP pour {motif_name} n'est pas un TS d'ordre 1 parfait.")
        print("L'échantillonnage continue pour valider l'infrastructure informatique (PoC).")

    print(f"\n--- [AIMD] Génération thermodynamique sur les images de {motif_name} ---")
    for i, img in enumerate(images):
        engine.run_aimd_sampling(img, motif_name, i)
        
    return {
        "motif": motif_name, 
        "qc_passed": is_valid_ts, 
        "images_sampled": len(images)
    }

@job
def train_mace_mlip_job(config: dict, dft_outputs: list = None):
    engine = SACPipelineEngine(config)
    n_train, n_val = engine.export_mace_datasets()
    print(f"Dataset exporté : {n_train} train, {n_val} val. Lancement de l'entraînement MACE...")
    
    trained_model_path = engine.train_mace_model()
    return {"model_path": trained_model_path}

@job
def evaluate_mace_model_job(model_path: str, config: dict):
    from mace.calculators import MACECalculator
    
    device = config.get("mace_mlip", {}).get("device", "cpu")
    calc = MACECalculator(model_paths=model_path, device=device)
    db = connect(config["system"]["db_path"])
    os.makedirs("figures", exist_ok=True)
    
    val_file = config["system"]["export_val_xyz"]
    val_atoms = read(val_file, index=":")
    
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

    print("\n--- Évaluation de la précision chimique (ΔE ads) ---")
    dft_de, mace_de, motif_labels = [], [], []
    
    all_rows = list(db.select())
    motifs = list(set([row.motif for row in all_rows if hasattr(row, 'motif')]))
    
    for motif in motifs:
        is_rows = list(db.select(motif=motif, stage="rattled_pathway_img0"))
        fs_rows = [r for r in all_rows if r.motif == motif and "img" in r.stage]
        if not is_rows or not fs_rows:
            continue
            
        fs_rows.sort(key=lambda x: x.stage)
        fs_row = fs_rows[-1]
        is_row = is_rows[0]
        
        dft_de.append(fs_row.energy - is_row.energy)
        
        atoms_is = is_row.toatoms()
        atoms_is.calc = calc
        mace_e_is = atoms_is.get_potential_energy()
        
        atoms_fs = fs_row.toatoms()
        atoms_fs.calc = calc
        mace_e_fs = atoms_fs.get_potential_energy()
        
        mace_de.append(mace_e_fs - mace_e_is)
        motif_labels.append(motif)
        
    if dft_de:
        df_ads = pd.DataFrame({"DFT_dE": dft_de, "MACE_dE": mace_de, "Motif": motif_labels})
        mae_ads = (df_ads["DFT_dE"] - df_ads["MACE_dE"]).abs().mean()
        print(f"MAE sur les énergies d'adsorption : {mae_ads:.4f} eV")
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        pmv.density_scatter(df_ads["DFT_dE"], df_ads["MACE_dE"], ax=ax)
        
        min_ads = min(df_ads.min(numeric_only=True)) - 0.2
        max_ads = max(df_ads.max(numeric_only=True)) + 0.2
        ax.plot([min_ads, max_ads], [min_ads, max_ads], 'k--', alpha=0.7, label="Parité parfaite")
        
        ax.set_xlabel(r"$\Delta E_{ads}$ DFT (eV)", fontweight="bold")
        ax.set_ylabel(r"$\Delta E_{ads}$ MACE (eV)", fontweight="bold")
        ax.set_title(f"H2 Adsorption Energy (MAE = {mae_ads:.3f} eV)", fontweight="bold")
        ax.legend()
        plt.savefig("figures/mace_adsorption_parity.png")
        plt.close()
        
    return {"mae_total": float(mae_tot), "mae_adsorption": float(mae_ads) if dft_de else None}

@job
def run_mlip_cineb_job(motif_name: str, structure_file: str, model_path: str, config: dict):
    atoms = read(structure_file)
    engine = SACPipelineEngine(config)
    
    metal = motif_name.split('-')[0]
    engine.sys_cfg['metal'] = metal
    
    motifs_dict = {motif_name: atoms}
    neb_results = engine.run_cineb_screening(motifs_dict, model_path)
    
    res = neb_results[motif_name]
    
    os.makedirs("structures_neb", exist_ok=True)
    write(f"structures_neb/{motif_name}_IS.xyz", res["initial_state"])
    write(f"structures_neb/{motif_name}_TS.xyz", res["transition_state"])
    write(f"structures_neb/{motif_name}_FS.xyz", res["final_state"])
    
    max_f = 0.0
    try:
        max_f = np.linalg.norm(res["transition_state"].get_forces(), axis=1).max()
    except:
        max_f = 99.0 
        
    return {
        "motif": motif_name,
        "e_act": res["e_act"],
        "delta_e": res["delta_e"],
        "pathway_type": res["pathway_type"],
        "energies": res["energies"], 
        "max_force": float(max_f)
    }

@job
def analyze_bep_scaling_job(neb_results_list: list):
    os.makedirs("figures", exist_ok=True)
    
    print("\n=======================================================")
    print(" RÉCAPITULATIF THERMODYNAMIQUE ET CINÉTIQUE (MACE)")
    print("=======================================================")
    print(f"{'Motif':<12} | {'E_act (eV)':<12} | {'Delta_E (eV)':<12}")
    print("-" * 55)
    
    fig_prof, ax_prof = plt.subplots(figsize=(8, 6), dpi=300)
    
    ea = []
    de = []
    valid_results = []
    
    for r in neb_results_list:
        motif = r["motif"]
        e_a = r["e_act"]
        d_e = r["delta_e"]
        energies = r.get("energies", [])
        
        print(f"{motif:<12} | {e_a:<12.3f} | {d_e:<12.3f}")
        
        if e_a > 0.05 and e_a < 5.0:
            ea.append(e_a)
            de.append(d_e)
            valid_results.append(r)
            
        if energies:
            e_norm = np.array(energies) - energies[0]
            x = np.linspace(0, 1, len(e_norm))
            ax_prof.plot(x, e_norm, marker='o', lw=2, label=f"{motif} (Ea={e_a:.2f} eV)")
            
    ax_prof.set_xlabel("Coordonnée de réaction normalisée", fontweight="bold")
    ax_prof.set_ylabel("Énergie relative (eV)", fontweight="bold")
    ax_prof.set_title("Profils d'énergie de dissociation H2 (CI-NEB)", fontweight="bold")
    ax_prof.legend()
    plt.tight_layout()
    plt.savefig("figures/neb_energy_profiles.png")
    plt.close()
    
    ea = np.array(ea)
    de = np.array(de)
    
    if len(ea) > 1:
        reg = LinearRegression().fit(de.reshape(-1, 1), ea)
        r2 = reg.score(de.reshape(-1, 1), ea)
        
        print(f"\n Équation BEP : Ea = {reg.coef_[0]:.2f} * dE + {reg.intercept_:.2f}")
        print(f" R² obtenu    : {r2:.3f}")
        print("=======================================================\n")
        
        fig_bep, ax_bep = plt.subplots(figsize=(7, 6), dpi=300)
        ax_bep.scatter(de, ea, color="#1f77b4", s=60, edgecolors="k", label="MACE Predictions")
        
        x_line = np.linspace(min(de), max(de), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax_bep.plot(x_line, y_line, "r--", label=f"Fit ($R^2$ = {r2:.2f})")
        
        for r in valid_results:
            ax_bep.annotate(r["motif"], (r["delta_e"], r["e_act"]), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax_bep.set_xlabel(r"Reaction Energy $\Delta E$ (eV)", fontsize=11, fontweight="bold")
        ax_bep.set_ylabel(r"Activation Barrier $E_a$ (eV)", fontsize=11, fontweight="bold")
        ax_bep.set_title("Brønsted-Evans-Polanyi Scaling Relation", fontsize=12, fontweight="bold")
        ax_bep.legend(frameon=True)
        
        plt.tight_layout()
        plt.savefig("figures/bep_scaling_validation.png")
        plt.close()
        
        return {"r2_score": float(r2), "slope": float(reg.coef_[0])}
    return {}

@job
def active_learning_decision_job(screening_results: list, config: dict):
    from jobflow import Flow, Response
    new_jobs = []
    motifs_to_recalculate = []

    # Extraction des seuils définis dans config.yaml
    max_ea = config.get("active_learning", {}).get("max_ea", 2.50)
    min_ea = config.get("active_learning", {}).get("min_ea", -0.05)
    fmax_tol = config.get("active_learning", {}).get("force_tol", 0.10)

    print("\n--- Évaluation de la robustesse du modèle MLIP ---")
    for data in screening_results:
        motif = data['motif']
        e_act = data['e_act']
        max_force = data.get('max_force', 99.0) 
        
        if motif.endswith("N3"):
            continue
            
        # Détection d'anomalies basée sur les paramètres utilisateur
        if e_act > max_ea or e_act <= min_ea or max_force > fmax_tol:
            print(f"⚠️️ Anomalie détectée sur {motif} : E_act = {e_act:.2f} eV, Max Force = {max_force:.2f} eV/Å")
            motifs_to_recalculate.append(motif)
        else:
            print(f"✅ {motif} validé par le MLIP (E_act = {e_act:.2f} eV)")

    if not motifs_to_recalculate:
        print("\n✅ Le modèle MACE est confiant sur tous les motifs. Fin de la boucle Active Learning.")
        bep_job = analyze_bep_scaling_job(screening_results)
        return Response(addition=Flow([bep_job], name="Final_Analysis"))

    print(f"-> Déclenchement de l'AIMD pour : {motifs_to_recalculate}")
    dft_jobs = []
    for motif in motifs_to_recalculate:
        structure_file = f"structures_3fold/structure_{motif}.xyz"
        # On force explicitement le calcul DFT en contournant le Smart Resume pour l'apprentissage actif
        dft_job = run_dft_reference_job(motif, structure_file, config, is_active_learning=True)
        dft_jobs.append(dft_job)
        new_jobs.append(dft_job)
    
    dft_outputs = [d.output for d in dft_jobs]
    train_job = train_mace_mlip_job(config, dft_outputs=dft_outputs)
    new_jobs.append(train_job)
    
    screening_jobs = []
    all_motifs = [data['motif'] for data in screening_results]
    for motif in all_motifs:
        structure_file = f"structures_3fold/structure_{motif}.xyz"
        screen_job = run_mlip_cineb_job(motif, structure_file, train_job.output["model_path"], config)
        screening_jobs.append(screen_job)
        new_jobs.append(screen_job)

    next_decision = active_learning_decision_job([j.output for j in screening_jobs], config)
    new_jobs.append(next_decision)

    al_flow = Flow(new_jobs, name="Active_Learning_Loop")
    return Response(replace=al_flow)