`apcmodels` is a Python library to implement computational models of the auditory system, simulate responses for these models to a range of stimuli used in psychophysical experiments, and analyze the outputs of these models. `apcmodels` utilizes code from a range of existing Python packages in the auditory modeling domain and provides a single unified API to access these models. 

logic of how to construct batches
- Start with an empty dict
- Use wiggle() to create desired combinations of parameters to test
- Use flatten() to flatten the constructed parameters list 
- Use repeat() to specify how many repetitions you want simulated for each combination of parameters
- Use evaluate() to evaluate any random variables in the parameters
- Use increment() to specify any incremented parameters to test at the bottom level

