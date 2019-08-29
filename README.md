# vine-local

Vine-local is a neural graph-based [verbal MWE identification](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task) tool built with Keras (with TensorFlow as backend). It served as a baseline model for the [vine](https://github.com/kawu/vine) model described in [1].

## Requirements

* Python 3.6
* Keras 2.2.4
* tensorflow 1.14
* pandas 0.24.2
* networkx 2.3

## Data format

Vine-local works with the PARSEME [.cupt](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_45_Format_specification) (extended CoNLL-U) format.

## Training, prediction and tagging process

__Note__: The script handles the training, prediction and tagging process in one run.

Vine-local requires training one model per VMWE category. See the [PARSEME annotation guidelines](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/) for a description of the different categories employed in the PARSEME annotation framework.

It takes the following as input:

* The train, test and dev sets of a [parseme dataset](https://gitlab.com/parseme/sharedtask-data/)
* Embedding file(s)
* The config file that specifies the experimental parameters

Furthermore the VMWE category hast to be specified (e.g. `VID`).

Running the script:

```
python vine_local.py \
  --lang_dir path/to/parseme/language_dir \
  --train path/to/train.cupt \
  --dev path/to/dev.cupt \
  --test path/to/test.cupt \
  --mwe_type VMWE_TYPE \
  --embs path/to/embedding_dir
```

The script then trains the model and makes predictions for the dev and test set for which it generates tagged versions of the input file in .cupt format.

References
----------

[1] Jakub Waszczuk, Rafael Ehren, Regina Stodden, Laura Kallmeyer, *A Neural
Graph-based Approach to Verbal MWE Identification*, Proceedings of the Joint
Workshop on Multiword Expressions and WordNet (MWE-WN 2019), Florence, Italy,
2019 (https://www.aclweb.org/anthology/W19-5113)