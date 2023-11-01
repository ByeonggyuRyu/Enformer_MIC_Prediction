# Enformer_MIC_Prediction
Welcome to the _Enformer_MIC_Prediction_ repository. This computational framework has been tailored to predict antibiotics MIC values of Klebsiella pneumoniae using genomic data and antibiotic molecular structures. This model adapts the Enformer architecture, initially conceptualized by [Avsec et al.](https://www.nature.com/articles/s41592-021-01252-x) for long-range genomic interaction prediction, to address the challenge of AMR prediction.
## Background
### Feature Engineering & Data Processing
**Extracting AMR Genes:** Focusing on the practical application, the Enformer_MIC_Model optimizes computational efficiency by targeting AMR genes. By utilizing the Resistance Gene Identifier (RGI) tool associated with CARD database, we extracted the AMR genes from each Kpn strains.

**Padding:** Sequences of 'N' character paddings representing unknown nucleotides are applied at both terminus and in between the AMR genes to standardized the length of input sequence as 98,304 bp.

### Encoding & Data Transformation
**Genomic Data Transformation:** Nucleotide bases 'A', 'C', 'G', 'T', and 'N' were respectively encoded as '[1, 0, 0, 0]', '[0, 1, 0, 0]', '[0, 0, 1, 0]', '[0, 0, 0, 1]', and '[0, 0, 0, 0]', resulting in the Genomic matrix of dimensions (98304, 4).

**Antibiotics Data Encoding:** We procured the isomeric SMILES data for antibiotics from the PubChem21 database. This data was transformed into a (130, 20) matrix using one-hot character encoding. The (130, 2) matrix was reshaped into (650, 4) and then vertically stacked to synchronize with the dimensions of the Genomic matrix.

### Final Data Preparation
**Combining & Labeling:** The encoding for each genome-antibiotic pair from our 32,309 samples was achieved by summing the Genomic and SMILES matrices and then undergoing a linear scaling operation (division by 2). Each pair was then labeled using the integer value equivalent to the Log2 of the laboratory-derived MIC value. This transformed our challenge into a multi-class classification task. We explored two labeling techniques: a conventional one-hot encoding of the exact label and a soft labeling strategy that highlighted 1-tier accurate labels close to the precise label to differentiate them from inaccurate labels.
