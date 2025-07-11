# DynamicCoTracker3: Detect 3D points belonging to dynamic objects
Run for example
```
python online_tracker_demo.py --experiments_path /home/allegro/davide_ws/habitat-lab/FisherRF-active-mapping/experiments/GaussianSLAM/GdvgFV5R1Z5-results/MP3D --grid_size 20 --window_len 8
```

### Attribution

This project includes code adapted from [CoTracker](https://github.com/facebookresearch/co-tracker), developed by Karaev et al. If you use this work, please cite their paper:

```bibtex
@inproceedings{karaev24cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author    = {Nikita Karaev and Iurii Makarov and Jianyuan Wang and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {arXiv:2410.11831}},
  year      = {2024}
}