NER-datasets for Portuguese


# HAREM

[HAREM](https://www.linguateca.pt/HAREM/) was an evaluation contest for named entity recognition in Portuguese. There were two editions:

#### First HAREM

- **data**: [ColeccaoDouradaHAREM.txt](ColeccaoDouradaHAREM.txt)
- **cite**: ["HAREM: An Advanced NER Evaluation Contest for Portuguese"](SantosSecoCardosoVilelaLREC2006.pdf)

#### Second HAREM

- **data**:

  -  [CDSegundoHAREMclassico.xml](CDSegundoHAREMclassico.xml): named-entities
  - [CDSegundoHAREM_TEMPO.xml](CDSegundoHAREM_TEMPO.xml): named-entities and time expressions are grounded to a time/date structure.
  - [CDSegundoHAREMReRelEM.xml](CDSegundoHAREMReRelEM.xml): named-entities and relationships.


- **cite**: ["Second HAREM: Advancing the State of the Art of Named Entity Recognition in Portuguese"](FreitasetalLREC2010.pdf)


**NOTE:** the XML format might be painfull to parse, check the Paramopama corpus which includes
the HAREM data in CoNNL format.



WikiNER
=======
A NER-corpus based on exploration of inter-document links in Wikipedia.

- **data**: https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500
- **cite**: ["Learning multilingual named entity recognition from Wikipedia"](https://www.sciencedirect.com/science/article/pii/S0004370212000276)


Paramopama
==========
Extends the PtBR version of WikiNER corpus, revising incorrect assigned tags in order to improve corpus quality. In the experiments the authors also produced a CoNNL format version of the HAREM corpus, which made publicly available:

- **data**:

  - [corpus_First_HAREM.txt](corpus_First_HAREM.txt)
  - [corpus_miniHAREM.txt](corpus_miniHAREM.txt)
  - [corpus_Paramopama.txt](corpus_Paramopama.txt)  
  - [corpus_paramopama+second_harem.txt](corpus_paramopama+second_harem.txt)


- **cite**: ["Paramopama: a Brazilian-Portuguese Corpus for Named Entity Recognition"](Paramopama.pdf)



leNER-Br
========

- **data**:
  - [train.conll](train.conll)
  - [dev.conll](dev.conll)  
  - [test.conll](test.conll)


- **cite**: ["LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text"](luz_etal_propor2018.pdf)
- **web**: https://cic.unb.br/~teodecampos/LeNER-Br/


