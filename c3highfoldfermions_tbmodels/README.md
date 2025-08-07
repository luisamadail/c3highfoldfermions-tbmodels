# Tight-binding calculations for \( C_3 \) hihg-fold fermions in honeycomb lattices

This repository contains Python code used to produce plots and results in:

> "Exotic edge states of \( C_3 \) high-fold fermions in honeycomb lattices", [DOI:10.1103/PhysRevResearch.6.043262]
> "Landau level structure of multi-orbital C3 high-fold fermionss", [DOI:preprint]
> "Orbitronics in trigonal molecular crystals", [DOI:preprint]


These works explore multi-orbital tight-binding models on honeycomb lattices featuring orbitals that transform under the \( C_3 \) group. For the preparation of the orbital states we employ a full tight-binding hamiltonian for triangular nanographenes of arbitrary dimension (na,nb) and recover the zero mode solutions on the low-energy regime as our basis set for the reduced Hamiltonian. The number of orbitals in the multi-orbital reduced model (na+nb-2) depends on the dimension of the [na,nb]triangulene, for the ones studied in the papers:

> [2,2]T -> 2 orbitals per unit cell
> [2,3]T -> 3 orbitals per unit cell
> [3,3]T -> 4 orbitals per unit cell
> [4,4]T -> 6 orbitals per unit cell


In this code the user can find:

> comparisons between the full and reduced bandstructures; 
> edge properties in zigzag and armchair nanoribbon; 
> response with application of external electric and magnetic fields as well as orbital Zeeman and sublattice perturbations; 
> orbital properties in the 2D and ribbon TB Hamiltonians. 

## Structure
- `auxC3.py`: auxiliary definitions for Hamiltonian construction.
- `htest.py`: core Hamiltonians: 2D, full and reduced models.
- `plotbs.py`: plot definitions for bandstructure and states.
- `plotorb2D.py`: orbital analysis for 2D Hamiltonians
- `plotorbrib.py`: orbital analysis for ribbon Hamiltonians
- `main.ipynb`: usage demo

## How to Run

```bash
pip install -r requirements.txt

