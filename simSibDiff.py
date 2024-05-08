"""
```
install dependencies:
pip install numpy pandas loguru zarr dask
```
simulate children from previous generation parents under assortative mating
based on output from genParents.py
includes direct, maternal, paternal, imprinting and sibling effect fixed in V_true below -- change if needed
```
python simSibDiff.py --indir MC/ --outdir results/ --AM True --batch True --saveStats True --rho 0.4
```
options:
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
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from loguru import logger
import zarr
import dask.array as da
import pathlib


def mate(m_gt1, m_gt2, f_gt1, f_gt2, map_cm, rng):
    recomb1 = genRecomb(map_cm, rng)
    recomb2 = genRecomb(map_cm, rng)
    c_gt1 =  recomb1 * m_gt1  +  (1-recomb1) * m_gt2
    c_gt2 =  recomb2 * f_gt1  +  (1-recomb2) * f_gt2
    swap = False
    # randomly swap haplotypes
    ## all of the haplotype is swapped
    if rng.uniform() < 0.5:
        swap = True
        temp = c_gt1.copy()
        c_gt1 = c_gt2.copy()
        c_gt2 = temp.copy()
    return c_gt1, c_gt2, swap

## recombination according to genetic map
def genRecomb(map_cm, rng):
    # generate positions
    pos = rng.exponential(scale=1/0.01, size=14) ## hard-coded values from sim1000G
    # get maximum position from input
    maxpos = np.max(map_cm)
    # if needed, generate more positions
    while(np.sum(pos) < maxpos):
        pos = np.append(pos, rng.exponential(scale=1/0.01, size=10))
    if(np.sum(pos) < maxpos):
        logger.info("Not enough recombination events.")
        return
    # cumulative sum of positions
    pos = np.cumsum(pos)
    l = len(map_cm)
    # generate recombination vector with all true entries
    recombVector = np.ones(l, dtype=bool)
    # find nearest index of pos in map_cm
    indices = find_nearest(pos, map_cm)
    # switch recombination vector according to indices
    for i in range(len(indices)):
        if(indices[i] < l-1):
            recombVector[indices[i]:] = np.logical_not(recombVector[indices[i]:])
    # randomly switch recombination
    if rng.uniform() < 0.5:
        recombVector = np.logical_not(recombVector)
    return recombVector.astype(int)

## findInterval
def find_nearest(pos, map_cm):
    index = np.array([])
    for i in range(len(pos)):
        index = np.append(index, np.argmin(np.abs(map_cm-pos[i])))
    return index.astype(int)


def assort(y1, y2, rho, rng):
    sd = 0.01 #
    y1 += rng.normal(0, sd)
    y2 += rng.normal(0, sd)
    #logger.info(f"{np.sqrt(1-np.var(y1))=}, {np.sqrt(1-np.var(y2))=}")
    #check if y1 and y2 have all the same values
    if np.std(y1)==0 or np.std(y2)==0:
        # only sort
        logger.info("Std(y)==0")
        order1=y1.argsort()
        order2=y2.argsort()
    else:
        
        # https://blogs.sas.com/content/iml/2020/12/17/generate-correlated-vector.html
        # Given a vector, x, and a correlation, rho, find y such that corr(x,y) = rho
        # order y1
        order1 = y1.argsort()
        y1 = y1[order1]
        # # center y1, y2
        y1 = y1-y1.mean()
        y2 = y2-y2.mean()
        # make unit vectors
        u1 = y1/np.linalg.norm(y1)
        u2 = y2/np.linalg.norm(y2) # used as an initial guess of vector correlated to y1 with angle arccos(rho)
        # project u2 onto u1 and create vector orthogonal to projection
        z = np.dot(u2.T,u1)*u1
        z = u2 - z
        # create new y2 correlated to u1 with rho
        new_y2 = rho*u1 + np.sqrt(1-rho**2)*z
        # create order2
        order2 = np.arange(len(y2))
        order2 = order2[y2.argsort()]
        order2 = order2[new_y2.argsort()]
        #corr = np.corrcoef(y1,y2[order2])[0,1]
        #logger.info(f"{corr=}")

    return order1, order2

##############################################################################
def main(k, indir, nchild, nfam, batch, ngen, nsim, nvar, ncvar, AM, rho, outdir, saveStats, saveX):

    # random generator
    rng = np.random.default_rng()

    ## CHANGE HERE IF NEEDED
    # create child genetic effects
    # direct, maternal, paternal, imprinting, sibling
    V_true = np.array([[0.2,0.0,0.0,0.0,0.0],
                    [0.0,0.2,0.0,0.0,0.0],
                    [0.0,0.0,0.2,0.0,0.0],
                    [0.0,0.0,0.0,0.0,0.2],
                    [0.0,0.0,0.0,0.0,0.2]])
    
    ## read genetic map from file
    # pos  chr cM
    map = pd.read_csv("chr4/chr4.b37.gmap.gz", sep="\t")
    logger.info(f"{map=}")

    # if batch generation is true, number of families is dependent on generations
    ## need 2^(ngen-1) batches or groups with nfam families in each 
    if batch:
        groups = 2**(ngen-1)
        nfam*=groups
        logger.info(f"Avoid inbreeding when moving through generations. {nfam} families in {groups} groups will be generated.")
    # number of individuals per generation: 2 parents per family
    n = nfam*nchild

    # rows  = variants, columns = samples/individuals
    vcf = pd.read_csv(indir+"/vcf_"+str(nvar)+"snps.vcf.gz", compression='gzip', index_col=0)
    logger.info(f"{vcf=}")
    chrs = vcf.iloc[:,0]
    pos = vcf.iloc[:,1]
    ## total number of variants in dataset
    p=len(vcf)
    ## check that genetic map matches chromsome and positions in vcf file
    #if np.sum(np.isin(np.unique(chrs), np.unique(map.iloc[:,0]))==False) > 0:
    if np.sum(np.isin(np.unique(chrs), np.unique(map.iloc[:,1]))==False) > 0:
        # this means that (parts of) the vcf file is not contained in the genetic map
        logger.info("Error: Genetic map does not contain same chromosome as vcf file.")
        return
    #if np.min(pos) < np.min(map.iloc[:,1]):
    if np.min(pos) < np.min(map.iloc[:,0]):
        logger.info(f"Error: Genetic map does not cover the positions in the vcf file: {np.min(pos)=}, {np.min(map.iloc[:,1])=}.")
        return
    #if np.max(pos) > np.max(map.iloc[:,1]):
    if np.max(pos) > np.max(map.iloc[:,0]):
        logger.info(f"Error: Genetic map does not cover the positions in the vcf file: {np.max(pos)=}, {np.max(map.iloc[:,1])=}.")
        return
    # delete unused data
    del vcf

    # linear interpolation between map.cm and map.bp evaluated at pos from vcf file
    map_cm = np.interp(pos, map.iloc[:p,0], map.iloc[:p,2])
    #logger.info(f"{map_cm=}")
    logger.info(f"Analysing {nfam} families with {nchild} offspring for {p} variants; {n} total individuals per generation")

    ## effect size
    ## generate effect sizes: all 0 except mth effect! 
    ## effect is constant through generations and simulations
    m = np.array(range(0,ncvar))
    beta = np.zeros((p,k))
    beta[m] = rng.multivariate_normal(np.zeros(k), V_true/len(m), len(m))
    rng.shuffle(beta)
    logger.info(f"{beta[m]=}")

    ## start simulations
    logger.info("---------- Starting simulations -------------")
    # generate nsim replicates
    for s in range(nsim):

        ## read in files
        # mother genotpye
        z = zarr.open(indir+'/sim'+str(s+1)+'/m_gt1.zarr/', mode='r')
        m_gt1 = da.from_zarr(z).compute()
        z = zarr.open(indir+'/sim'+str(s+1)+'/m_gt2.zarr/', mode='r')
        m_gt2 = da.from_zarr(z).compute()
        # father genoytpe
        z = zarr.open(indir+'/sim'+str(s+1)+'/f_gt1.zarr/', mode='r')
        f_gt1 = da.from_zarr(z).compute()
        z = zarr.open(indir+'/sim'+str(s+1)+'/f_gt2.zarr/', mode='r')
        f_gt2 = da.from_zarr(z).compute()

        # results
        est_mean = np.zeros((ngen, k+1))
        est_mean_c1 = np.zeros((ngen, k+1))
        est_mean_c2 = np.zeros((ngen, k+1))
        est_var = np.zeros((ngen, k+1))
        est_var_c1 = np.zeros((ngen, k+1))
        est_var_c2 = np.zeros((ngen, k+1))
        est_corr = np.zeros((ngen, k*(k-1)//2*nchild))
        est_rhoX = np.zeros((ngen, len(m)))
        est_rho = np.zeros(ngen)
        est_maf = np.zeros((ngen, len(m)))
        est_mf = np.zeros((ngen,6*len(m)))
        est_gf = np.zeros((ngen,3*len(m)))
        theor_mean = np.zeros(ngen)
        theor_var = np.zeros((ngen,2)) if len(m)==1 else np.zeros((ngen,6))
        theor_mf = np.zeros((ngen,6*len(m)))
        theor_gf = np.zeros((ngen,6))
        theor_gf_pheno = np.zeros((ngen,3*len(m)))
        theor_gf_geno = np.zeros((ngen,3*len(m)))
        theor_corr = np.zeros((ngen, k*(k-1)//2*len(m)))
        est_mean_sibdiff = np.zeros((ngen, k+1))
        est_var_sibdiff = np.zeros((ngen, k+1))
        theor_mean_sibdiff = np.zeros(ngen)
        theor_var_sibdiff = np.zeros((ngen)) if len(m)==1 else np.zeros((ngen,3))

        # run through ngen generations
        for g in range(ngen):
            logger.info(f"---------- simulation {s} generation {g} ----------")
            # keep track of parent of origin in each generation: mother = 1, father = -1
            poo_gt1 = np.ones(n)
            poo_gt2 = -np.ones(n)
            ## children haplotypes
            c_gt1 = np.zeros((n,p))
            c_gt2 = np.zeros((n,p))
            # phenotype children with separate components
            y = np.zeros((n,k+1))
            if g==0:
                est_rho[g] = 0

            ## loop through families
            for i in range(nfam):                                
                # mate parents to create nchild offspring
                for j in range(nchild):
                    id = 2*i+j
                    #logger.info(f"{i=}, {id=}")
                    c_gt1[id], c_gt2[id], swap = mate(m_gt1[i], m_gt2[i], f_gt1[i], f_gt2[i], map_cm, rng) 
                    # keep track of haplotypes being swapped for imprinting
                    if swap:
                        poo_gt1[id] *=-1
                        poo_gt2[id] *=-1
            # end loop families

            # create genotypes from haplotypes
            m_gt = m_gt1+m_gt2
            f_gt = f_gt1+f_gt2
            c_gt = c_gt1+c_gt2
            # minor allele frequency
            maf=c_gt.mean(axis=0)/2
            logger.info(f"{maf[m]=}")
            # calculate imprinting
            i_gt = np.where(np.equal(c_gt, 1), c_gt1*poo_gt1.reshape(n,1)+c_gt2*poo_gt2.reshape(n,1), 0)
            #logger.info(f"{np.unique(i_gt)=}")
            logger.info(f"{i_gt.shape=}")
            # calculate genetic components and phenotypes for each child
            g_m = np.matmul(m_gt, beta[:,1])
            g_f = np.matmul(f_gt, beta[:,2])
            logger.info(f"{g_m.shape=}")
            logger.info(f"{g_f.shape=}")
            for j in range(nchild):
                g_d = np.matmul(c_gt[j::nchild], beta[:,0])
                g_i = np.matmul(i_gt[j::nchild], beta[:,3])
                z = nchild-1 if j==0 else j-1
                g_s = np.matmul(c_gt[z::nchild], beta[:,4])
                g_all = g_d + g_m + g_f + g_i + g_s
                y[j::nchild] = np.array([g_all, g_d, g_m, g_f, g_i, g_s]).T
                est_corr[g,0*nchild+j] = np.corrcoef(g_d, g_m)[0,1]
                est_corr[g,1*nchild+j] = np.corrcoef(g_d, g_f)[0,1]
                est_corr[g,2*nchild+j] = np.corrcoef(g_d, g_i)[0,1]
                est_corr[g,3*nchild+j] = np.corrcoef(g_d, g_s)[0,1]
                est_corr[g,4*nchild+j] = np.corrcoef(g_i, g_m)[0,1]
                est_corr[g,5*nchild+j] = np.corrcoef(g_i, g_f)[0,1]
                est_corr[g,6*nchild+j] = np.corrcoef(g_i, g_s)[0,1]
                est_corr[g,7*nchild+j] = np.corrcoef(g_s, g_m)[0,1]
                est_corr[g,8*nchild+j] = np.corrcoef(g_s, g_f)[0,1]
                est_corr[g,9*nchild+j] = np.corrcoef(g_m, g_f)[0,1]
            
            ############## estimation
            ## correlation between mother and father genotype
            for i in range(len(m)):
                est_rhoX[g, i] = np.corrcoef(m_gt[:,m[i]], f_gt[:,m[i]])[0,1]
            logger.info(f"{g=}, {est_rho[g]=}, {est_rhoX[g]=}")
            # mean
            est_mean[g] = np.mean(y, axis=0)
            est_mean_c1[g] = np.mean(y[0::nchild], axis=0)
            est_mean_c2[g] = np.mean(y[1::nchild], axis=0)
            # variance
            est_var[g] = np.var(y, axis=0)
            est_var_c1[g] = np.var(y[0::nchild], axis=0)
            est_var_c2[g] = np.var(y[1::nchild], axis=0)
            # maf
            est_maf[g] = maf[m]
            # genotype frequency and mating frequencies
            for i in range(len(m)):
                _, counts = np.unique(c_gt[:,m[i]], return_counts=True)
                logger.info(f"{m[i]=}, {counts/np.sum(counts)=}")
                est_gf[g, 3*i:3*(i+1)] = counts/np.sum(counts)
                # mating frequencies
                cmf00 = np.where((np.equal(m_gt[:,m[i]],0) & np.equal(f_gt[:,m[i]],0)), 1, 0)
                cmf11 = np.where((np.equal(m_gt[:,m[i]],1) & np.equal(f_gt[:,m[i]],1)), 1, 0)
                cmf22 = np.where((np.equal(m_gt[:,m[i]],2) & np.equal(f_gt[:,m[i]],2)), 1, 0)
                cmf01 = np.where((np.equal(m_gt[:,m[i]],0) & np.equal(f_gt[:,m[i]],1)), 1, 0)
                cmf02 = np.where((np.equal(m_gt[:,m[i]],0) & np.equal(f_gt[:,m[i]],2)), 1, 0)
                cmf12 = np.where((np.equal(m_gt[:,m[i]],1) & np.equal(f_gt[:,m[i]],2)), 1, 0)
                cmf10 = np.where((np.equal(m_gt[:,m[i]],1) & np.equal(f_gt[:,m[i]],0)), 1, 0)
                cmf20 = np.where((np.equal(m_gt[:,m[i]],2) & np.equal(f_gt[:,m[i]],0)), 1, 0)
                cmf21 = np.where((np.equal(m_gt[:,m[i]],2) & np.equal(f_gt[:,m[i]],1)), 1, 0)
                mf00 = np.sum(cmf00)/nfam
                mf01 = (np.sum(cmf01) + np.sum(cmf10))/nfam/2
                mf02 = (np.sum(cmf02) + np.sum(cmf20))/nfam/2
                mf11 = np.sum(cmf11)/nfam
                mf12 = (np.sum(cmf12)+np.sum(cmf21))/nfam/2
                mf22 = np.sum(cmf22)/nfam
                est_mf[g, 6*i:6*(i+1)] = np.array([mf00, mf11, mf22, mf01, mf02, mf12])
                logger.info(f"{est_mf[g, 6*i:6*(i+1)]=}")
            
            ############## calculation
            p1 = np.ones(len(m))-maf[m]
            p2 = maf[m]
            rpq = est_rho[g]*p1*p2
            rXpq = est_rhoX[g]*p1*p2
            # mean
            theor_mean[g] = np.sum(2*p2*(beta[m,0]+beta[m,1]+beta[m,2]+beta[m,4]))
            ## random mating
            ## loci
            var_loci = 2*p1*p2*(beta[m,0]**2+beta[m,1]**2+beta[m,2]**2+beta[m,3]**2+beta[m,4]**2
                                + beta[m,0]*beta[m,1]+beta[m,0]*beta[m,2]+beta[m,1]*beta[m,3]-beta[m,2]*beta[m,3]
                                + beta[m,0]*beta[m,4]+beta[m,1]*beta[m,4]+beta[m,2]*beta[m,4] )
            if len(m)==1:
                theor_var[g,0] = var_loci
            else:
                theor_var[g,1] = np.sum(var_loci) 
                ## covariance between loci
                theor_var[g,2] = 0
                for i in range(len(m)):
                    for j in range(i):
                        # use child 1
                        theor_var[g, 2] += 2*np.corrcoef(c_gt[0::nchild,m[i]], c_gt[0::nchild,m[j]])[0,1]*np.sqrt(var_loci[i])*np.sqrt(var_loci[j])
                        # add sibling effect
                        #theor_var[g, 2] += 2*np.corrcoef(c_gt[1::nchild,m[i]], c_gt[1::nchild,m[j]])[0,1]*np.sqrt(var_loci[i])*np.sqrt(var_loci[j])
                ## total
                theor_var[g,0] = theor_var[g,1] + theor_var[g,2] 
            ## assortative mating
            ## loci
            var_loci_am = (rXpq*(1+est_rhoX[g])*(beta[m,0]**2-beta[m,3]**2+beta[m,4]**2
                                                +2*beta[m,0]*beta[m,1]+2*beta[m,0]*beta[m,2]+2*beta[m,0]*beta[m,4]
                                                -2*beta[m,1]*beta[m,3]+2*beta[m,2]*beta[m,3]
                                                +2*beta[m,1]*beta[m,4]+2*beta[m,2]*beta[m,4]+4*beta[m,1]*beta[m,2])
                            + 2*rXpq*(beta[m,1]**2+beta[m,2]**2
                                +beta[m,0]*beta[m,1]+beta[m,0]*beta[m,2]
                                +beta[m,1]*beta[m,3]-beta[m,2]*beta[m,3]
                                +beta[m,0]*beta[m,4]+beta[m,1]*beta[m,4]+beta[m,2]*beta[m,4]))
            if len(m)==1:
                theor_var[g,1] = var_loci + var_loci_am
            else:
                var_loci_tot = var_loci + var_loci_am
                theor_var[g,4] = np.sum(var_loci_tot)
                theor_var[g, 5] = 0
                for i in range(len(m)):
                    for j in range(i):
                        # use both children for correlation
                        theor_var[g, 5] += 2*np.corrcoef(c_gt[0::nchild,m[i]], c_gt[0::nchild,m[j]])[0,1]*np.sqrt(var_loci_tot[i])*np.sqrt(var_loci_tot[j])
                        # add sibling effect
                        #theor_var[g, 5] += 2*np.corrcoef(c_gt[1::nchild,m[i]], c_gt[1::nchild,m[j]])[0,1]*np.sqrt(var_loci_tot[i])*np.sqrt(var_loci_tot[j])
                # total
                theor_var[g,3] = theor_var[g,4] + theor_var[g,5]
            logger.info(f"--------- Mean and variance ---------")   
            logger.info(f"{est_mean[g,0]=}")
            logger.info(f"{theor_mean[g]=}")
            logger.info(f"{est_var[g,0]=}")
            logger.info(f"{theor_var[g]=}")
            # set order to dm, df, di, mi, fi, mf
            order = [0,1,9,2,4,5,3,7,8,6]
            # factors to multiply with for each element: cm, cf, mf, ci = 0, mi, fi, cs, ms, fs, si
            factor1 = [2, 2, 0, 0, +2, -2, 2, 2, 2, 0]
            factor2 = [4, 4, 4, 0, +0, +0, 4, 4, 4, 0]
            factor3 = [2, 2, 4, 0, -2, +2, 2, 2, 2, 0]
            l = 0
            for i in range(1,k):
                for j in range(i):
                    for q in range(len(m)):
                        # correlation = cov/sqrt(sigma*sigma)
                        theor_corr[g,order[l]+k*(k-1)*q//2] = (factor1[l]*p1[q]*p2[q] + factor2[l]*rXpq[q] + factor3[l]*rXpq[q]*est_rhoX[g,q])*beta[m[q],i]*beta[m[q],j]
                    l+=1
            logger.info(f"--------- Correlations ---------")
            logger.info(f"{est_corr[g]=}")
            logger.info(f"{theor_corr[g]=}")

            ## genotype frequencies
            logger.info(f"--------- frequencies ---------")
            a = p1**2 + rXpq
            b = 2*p1*p2 - 2*rXpq
            c = p2**2 + rXpq
            S = est_rhoX[g]/(2*p1*p2*(1+est_rhoX[g]))
            d0 = -2*p2
            d1 = p1-p2
            d2 = 2*p1
            if len(m)==1:
                theor_gf[g, 0:3] = np.array([p1**2+rpq, 2*(p1*p2-rpq), p2**2+rpq]).T
                theor_gf[g, 3:6] = np.array([a, b, c]).T
            else:
                theor_gf_pheno[g] = np.array([p1**2+rpq, 2*(p1*p2-rpq), p2**2+rpq]).flatten(order="F")
                theor_gf_geno[g] = np.array([a, b, c]).flatten(order="F")
            theor_mf[g] = np.array([a**2*(1+S*d0**2), b**2*(1+S*d1**2), c**2*(1+S*d2**2), a*b*(1+S*d0*d1), a*c*(1+S*d0*d2), b*c*(1+S*d1*d2)]).flatten(order="F")
            logger.info(f"{np.sum(theor_mf[g], axis=0)=}")
            logger.info(f"{np.sum(est_mf[g], axis=0)=}")

            ### sibling differences
            ydiff = y[0::nchild] - y[1::nchild]
            logger.info(f"{ydiff.shape=}")
            est_mean_sibdiff[g] = np.mean(ydiff, axis=0)
            est_var_sibdiff[g] = np.var(ydiff, axis=0)
            theor_mean_sibdiff[g] = 0
            ## variance loci
            if len(m)==1:
                theor_var_sibdiff[g] = 2*p1*p2*(1-est_rhoX[g])*(beta[m,0]**2 + beta[m,3]**2 + beta[m,4]**2 - 2*beta[m,0]*beta[m,4])
            else:
                var_loci_diff=2*p1*p2*(1-est_rhoX[g])*(beta[m,0]**2 + beta[m,3]**2 + beta[m,4]**2 - 2*beta[m,0]*beta[m,4])
                theor_var_sibdiff[g,1] = np.sum(2*p1*p2*(1-est_rhoX[g])*(beta[m,0]**2 + beta[m,3]**2 + beta[m,4]**2 - 2*beta[m,0]*beta[m,4])) 
                theor_var_sibdiff[g, 2] = 0
                for i in range(len(m)):
                    for j in range(i):
                        ## difference
                        theor_var_sibdiff[g, 2] += 2*np.corrcoef((c_gt[0::nchild,m[i]]-c_gt[1::nchild,m[i]]), (c_gt[0::nchild,m[j]]-c_gt[1::nchild,m[j]]))[0,1]*np.sqrt(var_loci_diff[i])*np.sqrt(var_loci_diff[j])
                # total
                theor_var_sibdiff[g,0] = theor_var_sibdiff[g,1] + theor_var_sibdiff[g,2]
            logger.info(f"--------- Sibling differences ---------")
            logger.info(f"{est_var_sibdiff[g]=}")
            logger.info(f"{theor_var_sibdiff[g]=}")
            ### + covariance cancels in sibs
            ########################

            ### save first generation (RM) and last generation (AM)
            if (g==0 or g==ngen-1) and saveX:
                #make sure output directory exists 
                pathlib.Path(outdir+"/sim"+str(s+1)).mkdir(parents=True, exist_ok=True)
                # save betas
                if g==0:
                    np.savetxt(outdir+"/sim"+str(s+1)+"/true_betas.txt", beta)
                    np.savetxt(outdir+"/sim"+str(s+1)+"/true_V.txt", np.matmul(beta.T, beta))
                X = np.zeros((k*p,nfam))
                X[0::k] = c_gt[:,0::nchild]
                X[1::k] = m_gt
                X[2::k] = f_gt
                X[3::k] = i_gt[:,0::nchild]
                logger.info(f"{X=}")
                z = zarr.array(X.T, chunks=(None,1000))
                logger.info(f"{z.info=}")
                zarr.save(outdir+"/sim"+str(s+1)+"/genotype_child1_gen"+str(g+1)+".zarr", z)
                X[0::k] = c_gt[:,1::nchild]
                X[3::k] = i_gt[:,1::nchild]
                logger.info(f"{X=}")
                z = zarr.array(X.T, chunks=(None,1000))
                logger.info(f"{z.info=}")
                zarr.save(outdir+"/sim"+str(s+1)+"/genotype_child2_gen"+str(g+1)+".zarr", z)
                np.savetxt(outdir+"/sim"+str(s+1)+"/y_child1_gen"+str(g+1)+".txt", y[0::nchild])
                np.savetxt(outdir+"/sim"+str(s+1)+"/y_child2_gen"+str(g+1)+".txt", y[1::nchild])
                ## difference
                np.savetxt(outdir+"/sim"+str(s+1)+"/y_diff_gen"+str(g+1)+".txt", (y[0::nchild,0]-y[1::nchild,0]))
                Xdiff = np.zeros((2,nfam))
                X[0::k] = c_gt[:,0::nchild]-c_gt[:,1::nchild]
                X[1::k] = i_gt[:,0::nchild]
                logger.info(f"{Xdiff=}")
                z = zarr.array(Xdiff.T, chunks=(None,1000))
                logger.info(f"{z.info=}")
                zarr.save(outdir+"/sim"+str(s+1)+"/genotype_diff_gen"+str(g+1)+".zarr", z)

            ## use children as parents for next generation
            ## this is based on 2 children
            if g < ngen-1:
                # mothers = child1, father = child2
                m_gt1 = c_gt1[0::nchild].copy()
                m_gt2 = c_gt2[0::nchild].copy()
                f_gt1 = c_gt1[1::nchild].copy()
                f_gt2 = c_gt2[1::nchild].copy()
                y_m = y[0::nchild]
                y_f = y[1::nchild]
                ## re-sort parents based on groups to avoid inbreeding if batch == True
                # switch fathers of group 1-2, 3-4, 5-6, etc. to avoid inbreeding
                # merge groups for the next generation
                # number of groups is 2^(ngen-g)
                if batch:
                    ngroups = 2**(ngen-1-g)
                    gsize = nfam//ngroups
                    logger.info(f"{ngen=}, {g=}, {gsize=}, {ngroups=}")
                    # loop through groups
                    for l in range(0,ngroups,2):
                        #logger.info(f"{l=}, {l*gsize=}, {(l+1)*gsize=}, {(l+2)*gsize=}")
                        # first haplotype
                        temp=f_gt1[l*gsize:(l+1)*gsize].copy() #temp=group1
                        f_gt1[l*gsize:(l+1)*gsize]=f_gt1[(l+1)*gsize:(l+2)*gsize].copy()  #group1=group2
                        f_gt1[(l+1)*gsize:(l+2)*gsize]=temp.copy()  #group2=temp
                        # second haplotype
                        temp=f_gt2[l*gsize:(l+1)*gsize].copy() #temp=group1
                        f_gt2[l*gsize:(l+1)*gsize]=f_gt2[(l+1)*gsize:(l+2)*gsize].copy()  #group1=group2
                        f_gt2[(l+1)*gsize:(l+2)*gsize]=temp.copy()  #group2=temp

                    # introduce assortative mating - needs to be done within each group!
                    # # order parents according to phenotype
                    if AM:
                        logger.info(f"Simulation with assortative mating with correlation {rho=}.")
                        for l in range(ngroups):
                            # logger.info(f"{l=}, {l*gsize=}, {(l+1)*gsize=}")
                            order1, order2 = assort(y_m[l*gsize:(l+1)*gsize, 0], y_f[l*gsize:(l+1)*gsize,0], rho, rng)
                            m_gt1[l*gsize:(l+1)*gsize]=m_gt1[l*gsize+order1]
                            m_gt2[l*gsize:(l+1)*gsize]=m_gt2[l*gsize+order1]
                            f_gt1[l*gsize:(l+1)*gsize]=f_gt1[l*gsize+order2]
                            f_gt2[l*gsize:(l+1)*gsize]=f_gt2[l*gsize+order2]
                            y_m[l*gsize:(l+1)*gsize]=y_m[l*gsize+order1]
                            y_f[l*gsize:(l+1)*gsize]=y_f[l*gsize+order2]
                        est_rho[g+1] = np.corrcoef(y_m[:,0], y_f[:,0])[0,1]

                else:
                    ## random sorting which creates inbreeding when moving through generations
                    rand_index = np.random.permutation(f_gt1.shape[0])
                    logger.info(f"Randomly shuffle fathers. This will create inbreeding when generating several generations.")
                    f_gt1 = f_gt1[rand_index] 
                    f_gt2 = f_gt2[rand_index] 

        # end loop generations
        # save            
        if saveStats:
            #make sure output directory exists 
            pathlib.Path(outdir+"/sim"+str(s+1)).mkdir(parents=True, exist_ok=True)
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_mean.txt", est_mean, header="all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_mean_c1.txt", est_mean_c1, header="child1: all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_mean_c2.txt", est_mean_c2, header="child2: all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_var.txt", est_var, header="all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_var_c1.txt", est_var_c1, header="child1: all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_var_c2.txt", est_var_c2, header="child2: all, direct, maternal, paternal, imprinting for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_corr.txt", est_corr, header="dm, df, di, im, if, mf alternating for child 1 and child 2 for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/theor_mean.txt", theor_mean, header="calculated mean for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/theor_var.txt", theor_var, header="calculated variance for all generations: total, loci, covariance under RM; tot., loc., cov. under AM or RM/AM")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_maf.txt", est_maf, header="est. minor allele frequencies for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_gf.txt", est_gf, header="est. genotpye frequencies for all generations")
            if len(m) == 1:
                np.savetxt(outdir+"/sim"+str(s+1)+"/theor_gf.txt", theor_gf, header="calculated genotype frequencies for all generations (first 3 columns with phenotypic corr., last 3 with genotypic corr.)")
            else:
                np.savetxt(outdir+"/sim"+str(s+1)+"/theor_gf_phenoCorr.txt", theor_gf_pheno, header="calculated genotype frequencies with phenotypic corr. for all generations")
                np.savetxt(outdir+"/sim"+str(s+1)+"/theor_gf_genoCorr.txt", theor_gf_geno, header="calculated genotype frequencies with genotypic corr. for all generations")
            np.savetxt(outdir+"/sim"+str(s+1)+"/theor_mf.txt", theor_mf, header="calculated mating frequencies for all generations (XmXf) 00, 11, 22, 01, 02, 12, 10, 20, 21")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_mf.txt", est_mf, header="estimated mating frequency (XmXf) 00, 11, 22, 01, 02, 12, 10, 20, 21")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_rho.txt", est_rho, header="phenotypic correlation between parents")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_rhoX.txt", est_rhoX, header="genotypic correlation between parents")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_mean_sibdiff.txt", est_mean_sibdiff, header="estimated mean in sibling difference")
            np.savetxt(outdir+"/sim"+str(s+1)+"/est_var_sibdiff.txt", est_var_sibdiff, header="estimated variance in sibling difference")
            np.savetxt(outdir+"/sim"+str(s+1)+"/theor_mean_sibdiff.txt", theor_mean_sibdiff, header="calc. mean in sibling difference")
            np.savetxt(outdir+"/sim"+str(s+1)+"/theor_var_sibdiff.txt", theor_var_sibdiff, header="calc. variance in sibling difference")
        
    # end loop simulations
    
##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Foward in time simulation of families, focusing on sibling differences.')
    parser.add_argument('--indir', type=str, help='path to input directory', required=True)
    parser.add_argument('--nchild', type=int, default=2, help='number of children per parent pair (default=2)')
    parser.add_argument('--nfam', type=int, default=1000, help='number of families in each batch (default=1000); if batch=False, nfam corresponds to the total number of families')
    parser.add_argument('--batch', type=bool, default=False, help='mate in batches to avoid inbreeding (default=False)')
    parser.add_argument('--ngen', type=int, default=10, help='number of generations (default=10)')
    parser.add_argument('--nsim', type=int, default=10, help='number of simulations/replicates (default=10)')
    parser.add_argument('--nvar', type=int, default=100, help='number of variants (default=100)')
    parser.add_argument('--ncvar', type=int, default=100, help='number of causal variants (default=10)')
    parser.add_argument('--AM', type=bool, default=False, help='turn on assortative mating (default=False)')
    parser.add_argument('--rho', type=float, default=0.6, help='correlation for AM - only needed if AM=True (default=0.6)')
    parser.add_argument('--saveX', type=bool, default=False, help='save genotypes(default=False)')
    parser.add_argument('--saveStats', type=bool, default=False, help='save genotypes(default=False)')
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
        k=5, # number of genetic components
        indir=args.indir,
        nchild=args.nchild,
        nfam=args.nfam,
        batch=args.batch,
        ngen=args.ngen,
        nsim=args.nsim,
        nvar=args.nvar,
        ncvar=args.ncvar,
        AM=args.AM,
        rho=args.rho,
        outdir=args.outdir,
        saveStats=args.saveStats,
        saveX=args.saveX,
        ) 
    logger.info("Done.")