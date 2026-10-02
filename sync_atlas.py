from maggma.stores import JSONStore
from pymongo import MongoClient
import pandas as pd

def sync_and_explore():
    print("Connexion à la base locale (JSON)...")
    local_store = JSONStore("jobflow_local_db.json")
    local_store.connect()
    
    docs_to_sync = list(local_store.query())
    
    uri = "mongodb+srv://safouanziat_db_user:PnxN5NCgk7NAfK5b@sacpipelinecluster.3xffyti.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri)
    db = client["sac_multitm_db"]
    collection = db["outputs"]

    if docs_to_sync:
        print(f"Synchronisation de {len(docs_to_sync)} documents vers MongoDB Atlas...")
        for doc in docs_to_sync:
            # Jobflow utilise le champ 'uuid' comme identifiant unique
            doc_id = doc.get("uuid", doc.get("_id"))
            # replace_one avec upsert=True évite les doublons si tu relances le script
            collection.replace_one({"uuid": doc_id}, doc, upsert=True)
        print("[✓] Synchronisation réussie.")
    else:
        print("[!] Aucune donnée locale trouvée à synchroniser.")

    print("\nExtraction des barrières d'activation depuis Atlas...")
    
    pipeline = [
        {"$match": {"output.e_act": {"$exists": True}}},
        {"$project": {
            "_id": 0, 
            "motif": "$output.motif", 
            "Ea (eV)": "$output.e_act", 
            "Delta E (eV)": "$output.delta_e",
            "Chemin": "$output.pathway_type"
        }},
        {"$sort": {"Ea (eV)": 1}}
    ]
    
    results = list(collection.aggregate(pipeline))
    if results:
        df = pd.DataFrame(results)
        print("\nTableau de bord cinétique actuel :")
        print(df.to_string(index=False))
    else:
        print("\nAucun résultat cinétique trouvé dans la base Cloud.")

if __name__ == "__main__":
    sync_and_explore()