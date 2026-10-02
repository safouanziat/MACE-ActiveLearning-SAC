from ase.db import connect
from pymongo import MongoClient

def sync_catalysis_to_mongo():
    print("Lecture de la base locale catalysis.db...")
    db = connect("catalysis.db")
    rows = list(db.select())
    
    if not rows:
        print("[!] La base catalysis.db est vide ou introuvable.")
        return
        
    print(f"Connexion à MongoDB Atlas... ({len(rows)} structures trouvées)")
    uri = "mongodb+srv://safouanziat_db_user:PnxN5NCgk7NAfK5b@sacpipelinecluster.3xffyti.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri)
    
    # Collection dédiée aux structures quantiques DFT brutes
    collection = client["sac_multitm_db"]["dft_structures"] 
    
    for row in rows:
        doc = {
            "ase_id": row.id,
            "motif": row.get("motif", "N/A"),
            "stage": row.get("stage", "N/A"),
            "energy_eV": row.energy,
            "forces_eV_A": row.forces.tolist() if row.forces is not None else None,
            "positions_A": row.positions.tolist(),
            "chemical_symbols": row.symbols,
            "converged": row.get("converged", True)
        }
        collection.replace_one({"ase_id": row.id}, doc, upsert=True)
        
    print("[✓] Synchronisation des structures quantiques réussie vers MongoDB Atlas.")

if __name__ == "__main__":
    sync_catalysis_to_mongo()