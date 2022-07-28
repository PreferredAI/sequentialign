# sequentialign
This is the Python implementation of the 'Sequentialign' algorithm in the AIED 2022 paper "Towards Aligning Slides and Video Snippets: Mitigating Sequence and Content Mismatches" by Ziyuan Liu and Hady W. Lauw. 

## Implementation Environment

- python == 3.8.8
- numpy==1.20.1
- scipy==1.6.2
- python-pptx==0.6.18

## Run

`python main.py`

### Parameter Setting
- -dn: dataset name
- -ev: eval_only: yes=evaluate metrics using cached results, no=run algorithms and overwrite cached results
- -th: filter_threshold: integer(s). indicates thresholds for the sequentialign irrelevant content filter

## Data 

Each dataset should be placed as a separate subdirectory in the `./data` folder. The name of a dataset is the name of the subdirectory in which it is contained. Each slide deck-video pair should occupy a sequentially-numbered subdirectory in the respective dataset directory, containing the following files: 

- slides.pptx: a .pptx PowerPoint file containing the deck of slides. 
- annotation.txt: a text file containing the duration of the video (in seconds) in its first line, and the start and end times of each slide in the video in subsequent lines. 
- transcript.json: a .json file containing the transcript of the video in YouTube format. 

A sample is provided in the `./data/sample` directory to illustrate this. 

## Reference
If you use our paper, including the code contained herein, please cite

```
@inproceedings{sequentialign,
  title={Towards Aligning Slides and Video Snippets: Mitigating Sequence and Content Mismatches},
  author={Ziyuan, Liu and Lauw, Hady W.}, 
  booktitle={The 23rd International Conference on Artificial Intelligence in Education},
  year={2022}
}
```
