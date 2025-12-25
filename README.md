## Investigating generalization of SFT from a Bayesian perspective 

### Experiment Setup

mixture dataset -(pre-training)-> 
prior distribution learnt from prior weights -(sft)-> 
specific persona induced through sft

- personas represented as finetuned llms

- mixture dataset used to train a mixture persona

- log(P(sequence | persona)) generated and used to infer a prior probability distribution over personas

