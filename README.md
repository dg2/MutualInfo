# Mutual Information in Hadoop Streaming

### Basics
We use preprocessing routines which automaticaly discretise continuous features 
and bundle categorical features with too many values.

The actual MapReduce process is very simple: The mapper simply emits (feat_id, label, feat_value) 
triplets, where feat_id is the key for the sorting phase. Then, the reducer collects data that 
corresponds to a same feature into a block, and runs that block through the MI_analysis routine.

### Running the job
Configure the parameters in streaming.sh and run it:

    ./streaming.sh

## Testing locally the MapReduce code:
    
    cat subsample | ./feat_group_mapper.py | sort | ./MI_reducer.py > test
