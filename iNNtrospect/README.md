# iNNtrospect
iNNtrospect - a Deep Neural Network analysis toolbox

With iNNtrospect, you can better understand learned representations of Deep Neural Networks.

## Approach

![NAP and Topomap Method Overview](assets/images/method_overview.png)

## Supported Frameworks

Model inference is only supported with Tensorflow 2.X.  
Analysis and visualization is independent of the framework. It requires activations (and gradients) as numpy arrays.

## Usage (Tensorflow 2.X)

Examples of running the entire pipeline usages are provided in `examples/`.  

Preparation:

- Provide a model (Tensorflow)
- prepare a configuration dictionary or file (see `examples/configs/MNIST_MLP.json`)
- Prepare the data

Data needs to be stored in batches as numpy files, including a `corpus.csv` file of format:
`file name, group name, (optional) output index`

```
batch0000.npy,0,0
batch0001.npy,0,0
batch0002.npy,1,1
batch0003.npy,1,1
```

Steps of the Pipeline:
`<output>` directory is defined in the configuration file

1. `process_corpus_file()` performs model inference for each batch and stores activations (and gradients) for every batch.   
   -> `<output>/acts/` and `<output>/grads/`
2. `align_data()` (optional) aligns activations in `<output>/acts/` according to gradients in `<output>/grads/`.  
	-> `<output>/aligned/`
3. `compute_naps()` computes average activations of (aligned activations).  
   -> `<output>/naps/`
4. `compute_contrastive_naps()` (optional) contrasts NAPs between a given set of groups of interest.  
   -> `<output>/contrastive_naps/`
5. `compute_topomap_layout()` and `compute_topomap_activations()` compute a topographic layout and representative activations per layer.  
   -> `<output>/topomap_data/`
6. `plot_topomaps()` creates visualizations of the topographic maps according to `<output>/topomap_data/`.  
   -> `<output>/topomap_plots/`

## Usage (other frameworks)

You need to provide activations and gradients in the same structure as `process_corpus_file()` in Pipeline step 1.  
Also, several auxillary dictionaries need to be created, in the `<output>/` directory: `group_name_to_files.json`, `group_name_to_index.json`, `index_to_group_name.json`.  
All following steps are independent of the Deep Learning Framework.

## References

```
Valerie Krug; Raihan Kabir Ratul; Christopher Olson and Sebastian Stober.
Visualizing Deep Neural Networks with Topographic Activation Maps.
In: HHAI 2023: Augmenting Human Intellect. IOS Press, 2023. 138-152.
```