from ase.db import connect
from ase.io import write

def extract_neb_path():
    db = connect("catalysis.db")
    
    # Récupérer toutes les images associées au Pd-C3
    images = []
    for row in db.select(motif="Pd-C3"):
        if hasattr(row, 'stage') and "img" in row.stage:
            images.append((row.stage, row.toatoms()))
            
    if not images:
        print("Erreur : Aucune trajectoire NEB trouvée pour Pd-C3.")
        return

    # Trier les images dans l'ordre du chemin réactionnel (img0, img1, img2...)
    images.sort(key=lambda x: x[0])
    atoms_list = [img[1] for img in images]
    
    # Sauvegarder en format trajectoire ASE
    output_file = "Pd-C3_pathway.traj"
    write(output_file, atoms_list)
    print(f"Succès ! {len(atoms_list)} images extraites.")
    print(f"Tapez la commande suivante pour visualiser : ase gui {output_file}")

if __name__ == "__main__":
    extract_neb_path()