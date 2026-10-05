# Splits

`train_videos.txt` (1,610 videos) and `test_videos.txt` (290 videos) list every video used, one per line, in the format `<split>__<Class>__<VideoName>`.

They match the official UCF-Crime train/test partition (810 anomalous + 800 normal train; 140 anomalous + 150 normal test). The script writes these lists automatically to `outputs_frames/subset_train.txt` and `subset_test.txt` on every run.
