import yaml
import os
from jobflow import Flow, run_locally
from maggma.stores import JSONStore
from jobflow import JobStore
from project2_workflow import (
    generate_metal_motifs_job,
    run_dft_reference_job,
    train_mace_mlip_job,
    evaluate_mace_model_job,
    run_mlip_cineb_job,
    analyze_bep_scaling_job,
    active_learning_decision_job
)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    metals = config["system"]["metals"]
    stages = config["stages"]
    all_jobs = []
    
    suffixes = ["N3", "N2C1", "N1C2", "C3"]
    
    if stages.get("generate_motifs", False):
        for metal in metals:
            all_jobs.append(generate_metal_motifs_job(metal, config))
            
    # --- 1. LANCEMENT DE LA DFT ---
    dft_jobs = []
    if stages.get("static_dft", False):
        for metal in metals:
            motif_name = f"{metal}-N3"
            structure_file = f"structures_3fold/structure_{motif_name}.xyz"
            if os.path.exists(structure_file):
                d_job = run_dft_reference_job(motif_name, structure_file, config)
                dft_jobs.append(d_job)
                all_jobs.append(d_job)

    # --- 2. ENTRAÎNEMENT MACE (Lié à la DFT) ---
    train_job = None
    if stages.get("train_mace", False):
        dft_outputs = [d.output for d in dft_jobs]
        train_job = train_mace_mlip_job(config, dft_outputs=dft_outputs)
        all_jobs.append(train_job)
        
        eval_job = evaluate_mace_model_job(train_job.output["model_path"], config)
        all_jobs.append(eval_job)

    # --- 3. CINETIQUE ET ACTIVE LEARNING ---
    if stages.get("cineb_screening", False) and train_job is not None:
        neb_jobs = []
        for metal in metals:
            for suffix in suffixes:
                motif_name = f"{metal}-{suffix}"
                structure_file = f"structures_3fold/structure_{motif_name}.xyz"
                
                if os.path.exists(structure_file):
                    neb_job = run_mlip_cineb_job(
                        motif_name, 
                        structure_file, 
                        train_job.output["model_path"], 
                        config
                    )
                    all_jobs.append(neb_job)
                    neb_jobs.append(neb_job)
        
        if neb_jobs:
            neb_outputs = [j.output for j in neb_jobs]
            # L'analyse BEP finale est gérée de manière autonome par la boucle de décision
            al_job = active_learning_decision_job(neb_outputs, config)
            all_jobs.append(al_job)

    wf = Flow(all_jobs, name="Autonomous_MultiTM_SAC_Pipeline")
    print("[Jobflow] Lancement de l'orchestrateur avec boucle Active Learning...")
    
    # --- CORRECTION DU STORE ---
    docs_store = JSONStore("jobflow_local_db.json")
    store = JobStore(docs_store)  # Jobflow nécessite d'envelopper le store Maggma
    
    run_locally(wf, store=store, ensure_success=True)
    print("[Jobflow] Exécution terminée avec succès.")

if __name__ == "__main__":
    main()