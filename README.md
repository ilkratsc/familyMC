# Simulating families

This code generates
under AM and RM
number of children fixed to 2, no difference due to sex

The code is tested with python/3.12.

## 1. Set up python environment:
```
module load python/3.12.0
python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy pandas scipy loguru gzip zarr tqdm dask
```

## 2. Get code:
```
git clone https://github.com/medical-genomics-group/familyMC
```

## 3. Run code

### a.) generate first generation parents:
The haplotypes of the parents are generated based on randomly selected variants of chr4. This part of the code is based on https://github.com/adimitromanolakis/sim1000G.
genParents.py

### b.) forward in time simulation either with trios or sibling pairs
simFamilies.py
or simSibDiff.py

change V

## 4. Output
