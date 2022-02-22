# Dataset_Recommendation_using_ensembel_approach-CoAuthor-

## Introduction
This repository is the required dataset and python implementation of paper "**Recommending Scientific Datasets Using Author Networks in Ensemble Methods**" with authors **Xu Wang**, **Frank van Harmelen** and **Zhisheng Huang**.

## Requirement before running experiment
You should "pip" install followling library in your python evironment:
- [pybind11](https://pybind11.readthedocs.io/en/stable/index.html#)
- [pyHDT](https://callidon.github.io/pyHDT/) or [rdflib-hdt](https://github.com/RDFLib/rdflib-hdt)
- [numpy](https://numpy.org/)
- [gensim](https://radimrehurek.com/gensim/)
- [tqdm](https://tqdm.github.io/)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [networkx](https://networkx.org/)

## Dataset

The dataset you needed for our ensembel datset recommendation algorithm:
- MAKG coauthor RDF/HDT file [download link](https://surfdrive.surf.nl/files/index.php/s/ibrwDJNem6fLUdk)
- MAKG paper/dataset title RDF/HDT file [download link](https://surfdrive.surf.nl/files/index.php/s/ibrwDJNem6fLUdk)
- MAKG paper/dataset abstract RDF/HDT file [download link](https://surfdrive.surf.nl/files/index.php/s/ibrwDJNem6fLUdk)
- MAKG pretrained author-entity embedding [download link](https://makg.org/dumps/2020-06-19/makg-embeddings-2020-06-19.tar.bz2)
- Seed dataset/paper txt file one dataset per line
- Candidate dataset/paper txt file one dataset per line
- Gold standard link between seeds and candidates RDF/HDT file

## Sample experiment
41 [seed datasets](../seeds_sample.txt) and 116 [candidate datasets](../cands_sample.txt), with 117 [gold standard link](Standard_sample.hdt).

## License
This repository is licensed under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

The Microsoft Academic Knowledge Graph, the linked data description files, and the ontology are licensed under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/index.html).
