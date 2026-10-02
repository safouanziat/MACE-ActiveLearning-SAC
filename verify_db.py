from ase.db import connect

def inspect_database(db_path="catalysis.db"):
    db = connect(db_path)
    total_rows = len(db)
    print(f"\n=== Analyse de {db_path} ===")
    print(f"Total des structures stockées : {total_rows}\n")

    if total_rows == 0:
        print("La base de données est vide.")
        return

    # On inspecte un échantillon (les 5 premières et la toute dernière)
    indices_to_check = list(range(1, min(6, total_rows + 1)))
    if total_rows > 5:
        indices_to_check.append(total_rows)

    for row_id in indices_to_check:
        row = db.get(id=row_id)
        
        # ASE stocke les paramètres du calculateur dans un dictionnaire
        calc_params = getattr(row, "calculator_parameters", {})
        
        # GPAW sérialise le paramètre PW sous forme de dictionnaire dans 'mode'
        mode_info = calc_params.get("mode", "Non spécifié / Inconnu")
        xc_info = calc_params.get("xc", "Inconnu")
        
        print(f"ID {row.id:03d} | Motif: {row.get('motif', 'N/A')} | Stage: {row.get('stage', 'N/A')}")
        print(f"  -> Énergie : {row.energy:.4f} eV")
        print(f"  -> Fonctionnelle : {xc_info}")
        print(f"  -> Mode (Cutoff) : {mode_info}")
        print("-" * 50)

if __name__ == "__main__":
    inspect_database()