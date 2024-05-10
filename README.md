# Simulating families

This simple forward-in-time simulation generates genotypes and phenotypes of parents and children with realistic linkage disequilibrium (LD), assuming direct, indirect genetic and parent-of-origin effects, under random or assorative mating. The assortment is done by ordering the phenotpyes of the parents, up to a certain correlation $\rho$ which can be set in the simulations.

The code is tested with python/3.12.0.

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
The code is run in two steps: <br/>
a.) The haplotypes and genotypes of the first generation parents are produced based on randomly selected variants of chromosome 4. This part of the code is based on https://github.com/adimitromanolakis/sim1000G. The data of chromosome 4 is taken from the 1000 genomes project (\url{http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/}) using vcfrandomsample (\url{https://github.com/vcflib/vcflib#vcflib}) to downsample the data to make it more manageable. <br/>
b.) Children are produced based on the first generation parents using a genetic map. The simulation is moved forward through generations keeping the number of individuals constant. Assortative mating can be introduced in this step. The foucs in this step can either be on calculating population mean and variance of child-mother-father trios or of differences in sibling pairs.

At the beginning of each script, instructions on how to run the code and input parameters can be found.


### a.) Generate first generation parents with genParents.py:

Possible input parameters:
```
--nfam          number of families per batch (default=1000)
--ngen          number of generations (default=10)
--nsim          number of simulations (default=10)
--nvar          number of variants, randomly sampled from chr4 subset (default=100)
--outdir        path to output directory (required)
--randomdata    default=False; switch to true if haplotypes should be generated from a binomial with prob=0.5 instead of LD from chr4
```


### b.) Forward-in-time simulation either with trios (simFamilies.py) or sibling pairs (simSibDiff.py):
The variance of the effects V_true has to be changed within each script, if other values are needed.

Possible input parameters:
```
--indir         path to input directory where output from genParents.py is saved (required)
--outdir        path to output directory (required)
--nchild        number of children per parent pair (default=2) -- might cause troubles if set to another number
--nfam          number of families per batch (default=1000), needs ot match input to genParents.py
--ngen          number of generations (default=10), needs ot match input to genParents.py
--nsim          number of simulations (default=10), needs ot match input to genParents.py
--nvar          number of variants, randomly sampled from chr4 subset (default=100), needs ot match input to genParents.py
--ncvar         number of causal variants (default=10), will be selected randomly from nvar
--AM            turn on assortative mating (default=False)
--rho           set correlation between parents' phenotypes (default=0.2) - only needed if AM=True
--batch         mate in groups (batches) to avoid inbreeding, only needed if AM=True (default=False)
--saveStats     save mean and variances (default=False)
--saveX         save genotypes and phenotypes for first and last generation (default=False)
```

## 4. Output
