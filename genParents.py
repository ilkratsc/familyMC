"""
```
install dependencies:
pip install numpy pandas scipy loguru gzip zarr tqdm
```
generate first generation parents based on a random subset of chr4
number of individuals is calculated as 2**(ngen-1)*nfam -- this corresponds to two parents per family
```
python genParents.py --oudir MC/
```
options:
--nfam          number of families per batch (default=1000)
--ngen          number of generations (default=10)
--nsim          number of simulations (default=10)
--nvar          number of variants, randomly sampled from chr4 subset (default=100)
--outdir        path to output directory (required)
--randomdata    default=False; switch to true if haplotypes should be generated from a binomial with prob=0.5 instead of LD from chr4
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger
from tqdm import trange
import gzip
import zarr
import pathlib

# read in names in vcf file
def get_vcf_names(vcf_path):
    with gzip.open(vcf_path, "rt") as ifile:
          for line in ifile:
            if line.startswith("#CHROM"):
                  vcf_names = line.strip('#\n').split('\t')
                  break
    ifile.close()
    return vcf_names

# generate haplotypes
def genhap(p, ld, q, rng):
    gt = rng.multivariate_normal(np.zeros(p), ld)
    ## convert to 0,1 based on quantiles
    gt = np.where(gt > q, 0, 1)
    return gt

##############################################################################
def main(nchild, nfam, ngen, nsim, nvar, randomdata, outdir):

    # random generator
    rng = np.random.default_rng()

    ## need 2^(ngen-1) batches or groups with nfam families in each 
    groups = 2**(ngen-1)
    nfam*=groups
    # number of individuals per generation: 2 parents per family
    n = nfam*nchild
  
    ## create random data if boolean is true
    if randomdata:
        # number of variants
        p=nvar
        # draw from binomial
        gt1 = pd.DataFrame(data=rng.binomial(1, 0.5, size=(p,n)))
        gt2 = pd.DataFrame(data=rng.binomial(1, 0.5, size=(p,n)))
    ## read in vcf file
    else:
        names = get_vcf_names('chr4/vcf_chr4_subset_small.vcf.gz')
        vcf = pd.read_csv('chr4/vcf_chr4_subset_small.vcf.gz', compression='gzip', comment='#', delim_whitespace=True, header=None, names=names)
        logger.info(f"{vcf=}")
        # randomly sample only nvar variants - using a randomly generated index
        ind = np.sort(np.random.randint(0, len(vcf), nvar))
        vcf = vcf.loc[ind]
        logger.info(f"{vcf=}")
        #save vcf file to know which variants were used
        #make sure output directory exists 
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        vcf.to_csv(outdir+"/vcf_"+str(nvar)+"snps.vcf.gz", compression='gzip')
        ## genotype information data starts at column 9
        ## read genotype information 0|1
        gt1 = vcf.iloc[:,9:].apply(lambda x: x.str.split("|").str[0]).astype(int)
        gt2 = vcf.iloc[:,9:].apply(lambda x: x.str.split("|").str[1]).astype(int)        
        ## total number of variants in dataset
        p=len(vcf)
        del vcf, names
    logger.info(f"Creating {nfam} families with {nchild} offspring for {p} variants; {n} total individuals per generation")

    ## genotype
    gt = gt1+gt2
    logger.info(f"{gt.shape=}")
    ## calculate minor allele frequency
    maf = gt.mean(axis=1)/2
    logger.info(f"{maf=}")
    ## check that all frequencies are between 0 and 1 (choose only polymorphic indices)
    index = maf.iloc[np.where(maf==0)].index
    index = index.append(maf.iloc[np.where(maf==1)].index)
    if len(index) > 0:
        logger.info("Removing non-polymorphic indices.")
        maf = maf.drop(index)
        gt = gt.drop(index)
        gt1 = gt1.drop(index)
        gt2 = gt2.drop(index)
        p = len(gt)
    #quantiles based on qnorm() function in R -> scipy.stats.norm.ppf()
    q = norm.ppf(maf)
    #logger.info(f"{q=}")
    ## LD
    logger.info(f"{gt=}")
    ld = np.cov(gt.values)
    logger.info(f"{ld=}")
    ## gt is only used to create first LD matrix
    del gt1, gt2, gt

    ## start simulations
    logger.info("---------- Starting MC generation -------------")
    # generate nsim replicates
    for s in range(nsim):
        logger.info(f"---------- Simulation {s} -------------")
        
        ngt = ["m_gt1", "m_gt2", "f_gt1", "f_gt2"]
        # generate 4 haplotypes, two per parent
        for j in range(4):
            logger.info(f"{ngt[j]}")
            ## container
            gt = np.zeros((nfam,p))
            ## loop through families
            for i in trange(nfam, desc="Loop"):
                gt[i] = genhap(p, ld, q, rng)
        
            # save haplotypes as zarr
            #make sure output directory exists 
            pathlib.Path(outdir+"/sim"+str(s+1)).mkdir(parents=True, exist_ok=True)
            z = zarr.array(gt, chunks=(1000,None))
            zarr.save(outdir+"/sim"+str(s+1)+'/'+ngt[j]+'.zarr', z)
        logger.info(f"Sim {s} done.")   
    # end loop simulations
    

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulating first generation parents.')
    parser.add_argument('--nfam', type=int, default=1000, help='number of families in each batch (default=1000); if batch=False, nfam corresponds to the total number of families')
    parser.add_argument('--ngen', type=int, default=10, help='number of generations (default=10)')
    parser.add_argument('--nsim', type=int, default=10, help='number of simulations/replicates (default=10)')
    parser.add_argument('--nvar', type=int, default=100, help='number of variants (default=100)')
    parser.add_argument('--randomdata', type=bool, default=False, help='create random data (default=False)')
    parser.add_argument('--outdir', type=str, help='path to output directory')
    args = parser.parse_args()
    logger.info(args)

    logger.remove()
    logger.add(
        sys.stderr,
        backtrace=True,
        diagnose=True,
        colorize=True,
        level=str("debug").upper(),
    )
    np.set_printoptions(precision=6, suppress=True)
    main(
        nchild=2,
        nfam=args.nfam,
        ngen=args.ngen,
        nsim=args.nsim,
        nvar=args.nvar,
        randomdata=args.randomdata,
        outdir=args.outdir,
        ) 
    logger.info("Done.")