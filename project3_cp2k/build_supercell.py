from ase.build import graphene
from ase.io import write

# Génération d'une feuille de graphène de 24.6 Å x 24.6 Å
atoms = graphene(formula='C2', a=2.46, size=(10, 10, 1), vacuum=10.0)

write("graphene_supercell.xyz", atoms)

print(f"Supercellule générée : {len(atoms)} atomes de Carbone.")
print(f"Dimensions de la boîte (Å) : {atoms.cell.lengths()}")