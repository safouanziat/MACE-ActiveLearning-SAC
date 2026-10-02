"""
Core Engine: Synthesis, Sampling, AIMD, MLIP Training, Active Learning,
CI-NEB, Thermochemistry, and Matplotlib Figure Generation.
"""

import os
import sys
import glob
import re
import subprocess
import numpy as np
import itertools

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

from ase import Atoms, Atom, units
from ase.build import make_supercell
from ase.io import read, write, Trajectory
from ase.db import connect
from ase.optimize import BFGS, FIRE
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep.neb import NEB
from ase.vibrations import Vibrations
from ase.md.langevin import Langevin
from gpaw import GPAW, PW

try:
    from gpaw.dftd3 import DFTD3
    HAS_DFTD3 = True
except ImportError:
    HAS_DFTD3 = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class SACPipelineEngine:
    def __init__(self, config: dict):
        self.cfg = config
        self.sys_cfg = config["system"]
        self.ads_cfg = config["adsorbate"]
        self.dft_cfg = config["dft"]
        self.mace_cfg = config.get("mace_mlip", {})
        self.kin_cfg = config.get("kinetics_and_thermo", {})
        
        self.db = connect(self.sys_cfg["db_path"])
        self.use_d3 = self.dft_cfg.get("use_d3", False) and HAS_DFTD3
        os.makedirs("figures", exist_ok=True)

    def verify_site_topology(self, slab: Atoms, expected_coord: int = 3):
        metal_sym = self.sys_cfg["metal"]
        syms = np.array(slab.get_chemical_symbols())
        metal_idx = np.where(syms == metal_sym)[0][0]
        
        dists = slab.get_distances(metal_idx, range(len(slab)), mic=True)
        dists[metal_idx] = 999.0  
        
        neighbors = np.where(dists < 2.4)[0]
        n_neighbors = len(neighbors)
        
        if n_neighbors != expected_coord:
            raise ValueError(f"\n[ERREUR FATALE] Topologie invalide : Le {metal_sym} a {n_neighbors} voisins.")
        print(f"  [✓] Topologie validée : {metal_sym} coordonné à {expected_coord} atomes.")

    def synthesize_pristine_sac(self) -> Atoms:
        metal = self.sys_cfg["metal"]
        coord_elem = self.sys_cfg["coordination_element"]
        cell_size = self.sys_cfg.get("cell_size", [5, 5])
        vacuum = self.sys_cfg.get("vacuum", 10.0)

        a = 2.46
        c = 2.0 * vacuum
        unit_cell = Atoms(
            "C2",
            positions=[[0.0, 0.0, vacuum], [a / 2.0, a / (2.0 * np.sqrt(3)), vacuum]],
            cell=[[a, 0, 0], [a / 2.0, a * np.sqrt(3) / 2.0, 0], [0, 0, c]],
            pbc=[True, True, False]
        )
        P = [[cell_size[0], 0, 0], [0, cell_size[1], 0], [0, 0, 1]]
        slab = make_supercell(unit_cell, P)

        com = np.mean(slab.positions[:, :2], axis=0)
        dists = np.linalg.norm(slab.positions[:, :2] - com, axis=1)
        center_carbon_idx = np.argmin(dists)
        
        metal_pos = slab.positions[center_carbon_idx]
        del slab[center_carbon_idx]

        slab.append(metal)
        metal_idx = len(slab) - 1
        metal_pos[2] += 1.0
        slab.positions[metal_idx] = metal_pos

        dists_to_tm = [slab.get_distance(metal_idx, i) for i in range(len(slab) - 1)]
        coord_indices = np.argsort(dists_to_tm)[:3]  
        for idx in coord_indices:
            slab.symbols[idx] = coord_elem

        self.verify_site_topology(slab, expected_coord=3)

        log_label = f"gpaw_logs/prerelax_{metal}_{coord_elem}3"
        slab.calc = self.get_gpaw_calculator(log_label)
        opt = BFGS(slab, logfile=f"{log_label}_opt.log", trajectory=f"{log_label}.traj")
        opt.run(fmax=self.sys_cfg.get("pre_relax_fmax", 0.05))
        slab.calc = None

        write(f"synthesized_{metal}_{coord_elem}3_relaxed.xyz", slab)
        return slab

    def generate_motifs(self, base_slab: Atoms) -> dict:
        metal_sym = self.sys_cfg["metal"]
        coord_elem = self.sys_cfg["coordination_element"]
        syms = np.array(base_slab.get_chemical_symbols())
        metal_idx = np.where(syms == metal_sym)[0][0]

        all_dists = [(i, base_slab.get_distance(metal_idx, i)) for i in range(len(base_slab)) if i != metal_idx]
        all_dists.sort(key=lambda x: x[1])
        ring = [idx for idx, _ in all_dists[:3]]

        motifs = {}
        def make_motif(sub_syms, label):
            s = base_slab.copy()
            for idx, sym in zip(ring, sub_syms):
                s.symbols[idx] = sym
            motifs[label] = s

        make_motif([coord_elem]*3, f"{metal_sym}-{coord_elem}3")
        make_motif([coord_elem, coord_elem, "C"], f"{metal_sym}-{coord_elem}2C1")
        make_motif([coord_elem, "C", "C"], f"{metal_sym}-{coord_elem}1C2")
        make_motif(["C", "C", "C"], f"{metal_sym}-C3")

        return motifs

    def generate_training_pathway(self, motif_name: str, slab: Atoms):
        metal_idx = np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]
        metal_pos = slab.positions[metal_idx]
        
        IS = slab.copy()
        IS.append(Atom('H', position=metal_pos + np.array([0, 0, 2.3])))
        IS.append(Atom('H', position=metal_pos + np.array([0.74, 0, 2.3])))
        
        FS = slab.copy()
        distances = slab.get_distances(metal_idx, range(len(slab)), mic=True)
        neighbors = [i for i, d in enumerate(distances) if 0 < d < 2.4 and slab.symbols[i] in ['C', 'N']]
        
        is_n_rich = "N2" in motif_name or "N3" in motif_name
        
        if is_n_rich and motif_name != "Ni-N3":
            FS.append(Atom('H', position=metal_pos + np.array([-1.0, 0, 1.5])))
            FS.append(Atom('H', position=metal_pos + np.array([1.0, 0, 1.5])))
        else:
            neigh_pos = slab.positions[neighbors[0]]
            FS.append(Atom('H', position=metal_pos + np.array([0, 0, 1.5])))
            FS.append(Atom('H', position=neigh_pos + np.array([0, 0, 1.1])))
            
        images = [IS.copy()] + [IS.copy() for _ in range(self.kin_cfg.get("neb_images", 5))] + [FS.copy()]
        neb = NEB(images)
        neb.interpolate('idpp')
        
        return images

    def verify_ts_vibrations(self, ts_atoms: Atoms, log_label: str) -> bool:
        """
        Calcule la Hessienne partielle sur l'état de transition présumé.
        Fige le graphène lointain et cible la première couronne via un rayon de coupe de 2.8 Å.
        """
        metal_sym = self.sys_cfg["metal"]
        metal_idx = np.where(np.array(ts_atoms.get_chemical_symbols()) == metal_sym)[0][0]
        
        active_indices = []
        for atom in ts_atoms:
            # On garde systématiquement le métal, l'adsorbat (H) et les hétéroatomes (N)
            if atom.symbol in [metal_sym, 'H', 'N']:
                active_indices.append(atom.index)
            # Pour les carbones, on filtre strictement par distance radiale autour du métal (< 2.8 Å)
            elif atom.symbol == 'C':
                dist = ts_atoms.get_distance(metal_idx, atom.index, mic=True)
                if dist < 2.8:
                    active_indices.append(atom.index)
                
        print(f"  -> QC Vibrations : {len(active_indices)} atomes actifs sur {len(ts_atoms)} (filtrage local < 2.8 Å).")
        
        ts_atoms.calc = self.get_gpaw_calculator(log_label)
        vib = Vibrations(ts_atoms, indices=active_indices, name=f"vib_{log_label}")
        vib.run()
        
        energies = vib.get_energies()
        imag_freqs = [e for e in energies if np.isreal(e) == False or e < 0]
        ts_atoms.calc = None
        
        print(f"  -> Fréquences imaginaires trouvées : {len(imag_freqs)}")
        return len(imag_freqs) == 1

    def run_aimd_sampling(self, atoms: Atoms, motif: str, img_idx: int):
        """
        Chauffe une image du NEB pour générer des géométries thermodynamiquement 
        déformées et enrichir le dataset d'entraînement MACE.
        """
        log_label = f"gpaw_logs/aimd_{motif}_img{img_idx}"
        atoms.calc = self.get_gpaw_calculator(log_label)
        
        temp = self.cfg.get("thermal_sampling", {}).get("temperature_K", 500)
        steps = self.cfg.get("thermal_sampling", {}).get("md_steps", 3)
        save_freq = self.cfg.get("thermal_sampling", {}).get("save_every", 1)
        
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=temp, friction=0.01)
        
        def save_state():
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            meta = {
                "motif": motif, 
                "stage": f"aimd_img{img_idx}", 
                "converged": True,
                "temperature": temp
            }
            self.save_record(atoms.copy(), energy, forces, meta)
            
        dyn.attach(save_state, interval=save_freq)
        dyn.run(steps)
        atoms.calc = None

    def discover_final_states(self, motif_name: str, slab: Atoms, model_path: str):
        from mace.calculators import MACECalculator
        device = self.mace_cfg.get("device", "cpu")
        calc = MACECalculator(model_paths=model_path, device=device)
        
        metal_idx = np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]
        metal_pos = slab.positions[metal_idx]
        
        num_samples = self.kin_cfg.get("discovery_samples", 30)
        best_e = float('inf')
        best_atoms = None
        
        for _ in range(num_samples):
            test_slab = slab.copy()
            test_slab.calc = calc
            for _ in range(2):
                r = np.random.uniform(1.2, 3.0)
                theta = np.random.uniform(0, np.pi/2)
                phi = np.random.uniform(0, 2*np.pi)
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                test_slab.append(Atom('H', position=metal_pos + np.array([x, y, z])))
            
            try:
                opt = BFGS(test_slab, logfile=None)
                opt.run(fmax=self.kin_cfg.get("neb_fmax", 0.05), steps=80)
                e = test_slab.get_potential_energy()
                
                syms = np.array(test_slab.get_chemical_symbols())
                h_indices = np.where(syms == "H")[0]
                d_hh = test_slab.get_distance(h_indices[0], h_indices[1], mic=True)
                
                if d_hh > 1.0 and e < best_e:
                    best_e = e
                    best_atoms = test_slab.copy()
            except Exception:
                continue
                
        return best_atoms, best_e

    def get_gpaw_calculator(self, log_label: str) -> GPAW:
        setups = {}
        if self.dft_cfg.get("hubbard_u", 0.0) > 0.0:
            setups[self.sys_cfg["metal"]] = f":d,{self.dft_cfg['hubbard_u']:.2f},0.0"
        return GPAW(
            mode=PW(self.dft_cfg["pw_cutoff"]),
            xc=self.dft_cfg["xc"],
            setups=setups if setups else "paw",
            spinpol=self.dft_cfg["spinpol"],
            kpts={"size": tuple(self.dft_cfg["kpts"]), "gamma": True},
            convergence=self.dft_cfg["convergence"],
            txt=f"{log_label}.log"
        )

    def run_dft_evaluation(self, atoms: Atoms, log_label: str) -> tuple:
        atoms.calc = self.get_gpaw_calculator(log_label)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if self.use_d3:
            try:
                d3 = DFTD3(xc=self.dft_cfg["xc"], damping="bj")
                e_d3, f_d3 = d3.calculate(atoms)
                energy += e_d3
                forces += f_d3
            except Exception:
                pass
        atoms.calc = None
        return energy, forces

    def save_record(self, atoms: Atoms, energy: float, forces: np.ndarray, meta: dict) -> int:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return self.db.write(atoms, key_value_pairs=meta)

    def export_mace_datasets(self):
        records = [row.toatoms() for row in self.db.select(converged=True)]
        if not records:
            raise RuntimeError("No records found in database.")
        np.random.seed(42)
        idx = np.random.permutation(len(records))
        split = int(len(records) * self.sys_cfg.get("train_ratio", 0.85))
        train_atoms = [records[i] for i in idx[:split]]
        val_atoms = [records[i] for i in idx[split:]]
        write(self.sys_cfg["export_train_xyz"], train_atoms, format="extxyz")
        write(self.sys_cfg["export_val_xyz"], val_atoms, format="extxyz")
        return len(train_atoms), len(val_atoms)

    def train_mace_model(self) -> str:
        cmd = [
            "mace_run_train",
            f"--name={self.mace_cfg.get('model_name', 'mace_sac_model')}",
            f"--train_file={self.sys_cfg['export_train_xyz']}",
            f"--valid_file={self.sys_cfg['export_val_xyz']}",
            "--E0s=average",
            "--atomic_numbers=[1, 6, 7, 27, 28, 46]",
            "--energy_key=energy",
            "--forces_key=forces",
            f"--energy_weight={self.mace_cfg.get('energy_weight', 1.0)}",
            f"--forces_weight={self.mace_cfg.get('forces_weight', 10.0)}",
            "--model=ScaleShiftMACE",
            "--hidden_irreps=64x0e + 64x1o",
            "--r_max=5.0",
            f"--batch_size={self.mace_cfg.get('batch_size', 4)}",
            "--valid_batch_size=4",
            f"--max_num_epochs={self.mace_cfg.get('max_epochs', 60)}",
            "--ema",
            f"--device={self.mace_cfg.get('device', 'cpu')}",
            "--default_dtype=float32"
        ]
        log_path = "mace_training.log"
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"MACE training failed. Check {log_path} for details.")

        expected_model = f"{self.mace_cfg.get('model_name', 'mace_sac_model')}_stagetwo.model"
        if not os.path.exists(expected_model):
            expected_model = f"{self.mace_cfg.get('model_name', 'mace_sac_model')}.model"
        return expected_model

    def _build_dissociated_final_state(self, slab: Atoms, metal_idx: int, model_path: str, device: str, motif_name: str) -> tuple:
        from mace.calculators import MACECalculator
        mace_calculator = MACECalculator(model_paths=model_path, device=device)
        
        theoretical_state = slab.copy()
        theoretical_state.calc = mace_calculator
        metal_pos = theoretical_state.positions[metal_idx]
        distances = theoretical_state.get_distances(metal_idx, range(len(theoretical_state)), mic=True)
        neighbors = [i for i, d in enumerate(distances) if 0 < d < 2.4 and theoretical_state.symbols[i] in ['C', 'N']]
        
        is_n_rich = "N2" in motif_name or "N3" in motif_name
        if is_n_rich and motif_name != "Ni-N3":
            theoretical_state.append(Atom('H', position=metal_pos + np.array([-1.0, 0, 1.5])))
            theoretical_state.append(Atom('H', position=metal_pos + np.array([1.0, 0, 1.5])))
        else:
            neigh_pos = theoretical_state.positions[neighbors[0]]
            theoretical_state.append(Atom('H', position=metal_pos + np.array([0, 0, 1.5])))
            theoretical_state.append(Atom('H', position=neigh_pos + np.array([0, 0, 1.1])))
            
        opt = BFGS(theoretical_state, logfile=None)
        opt.run(fmax=self.kin_cfg.get("neb_fmax", 0.05), steps=80)
        theo_energy = theoretical_state.get_potential_energy()
        
        discovered_state, disc_energy = self.discover_final_states(motif_name, slab, model_path)
        
        if discovered_state is not None and disc_energy < (theo_energy - 0.05):
            return discovered_state, "Decouvert_MLIP"
        else:
            return theoretical_state, "Forcé_Théorique"

    def run_cineb_screening(self, motifs: dict, model_path: str) -> dict:
        from mace.calculators import MACECalculator
        mol_type = self.ads_cfg["type"]
        d_eq = self.ads_cfg["eq_bond_length"]
        device = self.mace_cfg.get("device", "cpu")
        results = {}

        for motif_name, slab in motifs.items():
            print(f"  -> Running MACE CI-NEB on {motif_name}...")
            metal_idx = np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]
            metal_pos = slab.positions[metal_idx]

            initial = slab.copy()
            h1 = metal_pos + np.array([-d_eq / 2.0, 0.0, 2.3])
            h2 = metal_pos + np.array([d_eq / 2.0, 0.0, 2.3])
            initial.extend(Atoms(mol_type, positions=[h1, h2]))
            initial.calc = MACECalculator(model_paths=model_path, device=device)
            opt_i = BFGS(initial, logfile=None)
            opt_i.run(fmax=self.kin_cfg.get("neb_fmax", 0.05))

            final, pathway_type = self._build_dissociated_final_state(slab, metal_idx, model_path, device, motif_name)
            final.calc = MACECalculator(model_paths=model_path, device=device)

            n_images = self.kin_cfg.get("neb_images", 5)
            images = [initial]
            for _ in range(n_images):
                image = initial.copy()
                image.calc = MACECalculator(model_paths=model_path, device=device)
                images.append(image)
            images.append(final)

            neb = NEB(images, climb=self.kin_cfg.get("neb_climbing", True))
            neb.interpolate("idpp")
            
            fmax_target = self.kin_cfg.get("neb_fmax", 0.05)
            log_path = f"gpaw_logs/neb_{motif_name}.log"
            traj_path = f"neb_{motif_name}.traj"
            
            try:
                opt_neb = BFGS(neb, trajectory=traj_path, logfile=log_path)
                opt_neb.run(fmax=fmax_target, steps=800)
                if not opt_neb.converged():
                    raise RuntimeError("BFGS non convergé.")
            except Exception:
                opt_neb = FIRE(neb, trajectory=traj_path, logfile=log_path)
                opt_neb.run(fmax=fmax_target, steps=2500)

            energies = [img.get_potential_energy() for img in images]
            e_act = max(energies) - energies[0]
            delta_e = energies[-1] - energies[0]

            results[motif_name] = {
                "e_act": float(e_act),
                "delta_e": float(delta_e),
                "pathway_type": pathway_type,
                "energies": energies,
                "initial_state": initial,
                "transition_state": images[int(np.argmax(energies))],
                "final_state": final
            }
        return results