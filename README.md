# nomono-anonymizer - Face anonymizer on movie scenes

This program anonymizes with the Henohenomoheji (no-mo-no) letters by using the Keypoint R-CNN architecture.

## Usage
```
$ python nomono-anonymizer.py <input-video-file> <output-video-file>
```

## Generated videos
* `nomono-anonymizer.py` generates the Henohenomoheji (no-mo-no) letters on eyes and noses on movie scenes of these videos ([People in Italy](https://pixabay.com/videos/id-6582/) and [Guests at dinner](https://pixabay.com/videos/id-34418/)).

![People in Italy](images/nomono-anonymizer-italy.gif)

![Guests in the restaurant](images/nomono-anonymizer-dinner.gif)

## References
* https://github.com/kkroening/ffmpeg-python
* https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn
