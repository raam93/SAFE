
SAFE (**S**elf-**A**ssessing **F**aithfulness for **E**xplanations)
====================================================================

Repo structure:
---------------
* `input_data <https://github.com/SAFE/tree/master/input_data>`_ contains the datasets used in the experiments.
* `utils <https://github.com/SAFE/tree/master/utils>`_ contains files with helper functions.
* `data_loader.py <https://github.com/SAFE/tree/master/data_loader.py>`_ contains the class and methods required to load input texts (prompts) to LMs.
* `safe.py <https://github.com/SAFE/tree/master/safe.py>`_ is the main file that a user will run for experiments.

Datasets:
---------
In addition to the datasets provided in the `data <https://github.com/SAFE/tree/master/input_data>`_, we need to experiment on the following datasets. Add more if anything seems relevant.

There are two types of datasets:

**(Mostly) Independent texts**:
1. https://paperswithcode.com/dataset/civil-comments.  
    a. This is a large datasets, so a random sample of 10,000 is provided in the `data <https://github.com/SAFE/tree/master/input_data>`_  folder.

2. `Context Sensitivity Estimation in Toxicity Detection <https://aclanthology.org/2021.woah-1.15/>`_
    a. Code availability is not unclear - TODO -
    b. But this dataset is based on the above Civil Comments dataset, so we can easily reproduce.

3. `HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection <https://arxiv.org/abs/2012.10289>`_

4. `Designing Toxic Content Classification for a Diversity of Perspectives
 <https://arxiv.org/abs/2106.04511>`_


**Texts completed by LLMs** (these datasets will mostly have a human reference):
1. `Unveiling the Implicit Toxicity in Large Language Models <https://aclanthology.org/2023.emnlp-main.84/>`_
2. ToxiGen
3. RealToxicityPrompts
4. ...

To add a new dataset and related processing steps:
--------------------------------------------------
1. Add the dataset name and path in `data_path_map.json <https://github.com/SAFE/tree/master/utils/input_data_path_map.json>`_ (this is because input data can be huge and can exist in local)
    a. (Optionally) Add the dataset to the `data <https://github.com/SAFE/tree/master/input_data>`_ folder. **If the dataset is huge in size, skip uploading to master** using using `.gitignore <https://github.com/SAFE/tree/master/.gitignore>`_
2. Include the dataset metadata in `datasets_metadata.md <https://github.com/SAFE/tree/master/input_data/input_data_metadata.md>`_ (we will remove or reorganize this later)
3. Include the main processing function for the dataset in `data_processor.py <https://github.com/SAFE/tree/master/utils/data_processor.py>`_ and give it the same name as the dataset name.
4. Use "self." to access common parameters and functions of the `DataLoader` class in `data_loader.py <https://github.com/SAFE/tree/master/data_loader.py>`_.

### Tasks for Joanna:
---------------------
1. Check if combining prompt and response is useful for different datasets. If so, experiment and find out good strategies.
2. Try multiple prompts and concatenate with existing LLM-generated responses  
    a. Try different phrasing and one-shot/few-shot demonstrations.
3. For prompt-completion-type datasets, we will mostly have explicitly/implicitly toxic texts. 
    a. However, since we need toxic, non-toxic, and maybe toxic texts, we need to find a way to create non-toxic and maybe toxic texts with Toxigen-type datasets
    b. Perhaps we can use the original human responses (that are cut and used as prompts) as references?

### Tasks for Ram:
------------------
1. Start writing evaluator classes and methods for two examples and then scale it up.