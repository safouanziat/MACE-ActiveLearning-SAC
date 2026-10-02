import yaml
from ase.io import read
from pipeline_engine import SACPipelineEngine

# 1. Charger la configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- CORRECTIF : Forcer le métal pour ce script ---
if "system" in config:
    config["system"]["metal"] = "Pd"
else:
    config["metal"] = "Pd"
# --------------------------------------------------

# 2. Initialiser le moteur du pipeline
engine = SACPipelineEngine(config)

# 3. Charger la structure de base relaxée (déjà calculée par GPAW)
print("Chargement de la structure relaxée...")
base_slab = read("synthesized_Pd_N3_relaxed.xyz")

# 4. Générer les variations de la première couronne (motifs)
motifs = engine.generate_motifs(base_slab)

# 5. Lancer le screening CI-NEB avec le modèle MACE compilé
model_path = "mace_multitm_model_compiled.model"
print(f"Lancement du CI-NEB accéléré via {model_path}...")
results = engine.run_cineb_screening(motifs, model_path)

# 6. Afficher les barrières d'activation
print("\n--- RÉSULTATS DU SCREENING CINÉTIQUE ---")
for motif, data in results.items():
    print(f"Motif : {motif}")
    print(f"  Barrière d'activation (E_act) : {data['e_act']:.3f} eV")
    print(f"  Énergie de réaction (Delta_E) : {data['delta_e']:.3f} eV")
    print(f"  Mécanisme de l'état final   : {data['pathway_type']}")
    print("-" * 40)