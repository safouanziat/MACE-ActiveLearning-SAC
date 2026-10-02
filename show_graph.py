import yaml
from project2_workflow import generate_metal_motifs_job
from jobflow import Flow

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

metals = config["system"]["metals"]
all_jobs = [generate_metal_motifs_job(metal, config) for metal in metals]
wf = Flow(all_jobs, name="Autonomous_MultiTM_SAC_Pipeline")

# Utilise draw_graph pour afficher ou enregistrer le schéma
plt = wf.draw_graph()
plt.savefig("figures/workflow_dag.png", dpi=300, bbox_inches="tight")
print("Graphe sauvegardé avec succès dans figures/workflow_dag.png !")