# Chainer\_Realtime\_Multi-Person\_Pose\_Estimation

このリポジトリは [Realtime Multi-Person Pose Estimation](https://arxiv.org/abs/1611.08050) の論文をchainerで再現実装したものです。

<a href="README.md">English README</a>

## 実行結果

<p align="center">
<img src="data/movie_result.gif" width="720">
</p>


<div align="center">
<img src="data/people.png" width="300" height="300">
&nbsp;
<img src="data/people_result.png" width="300" height="300">
</div>


<p align="center">
<img src="data/demo_result.gif" width="620">
</p>

This project is licensed under the terms of the <a href="LICENSE">license</a>.

## コンテンツ
1. [caffemodel変換](#caffemodelをchainer用に変換)
2. [推論](#推論手順)
3. [訓練](#訓練手順)


## 環境

- Python 3.0+
- Chainer 2.0+
- NumPy
- Matplotlib
- OpenCV


## caffemodelをchainer用に変換
以下のコマンドで、訓練済みcaffemodelのダウンロード、npz形式への変換を行う。

```
cd models
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
python convert_model.py posenet pose_iter_440000.caffemodel coco_posenet.npz
python convert_model.py facenet pose_iter_116000.caffemodel facenet.npz
python convert_model.py handnet pose_iter_102000.caffemodel handnet.npz
cd ..
```

## 推論手順
以下のコマンドで、訓練済み重みパラメータファイルと画像を指定してポーズの推論を行う。処理結果は`result.png`という画像ファイルに保存される。

```
python pose_detector.py posenet models/coco_posenet.npz --img data/person.png 
```

GPUを使う場合は、--gpuオプションを付ける。

```
python pose_detector.py posenet models/coco_posenet.npz --img data/person.png --gpu 0
```

<div align="center">
<img src="data/person.png" width="300" height="300">
&nbsp;
<img src="data/person_result.png" width="300" height="300">
</div>

同様に、以下のコマンドで顔のランドマークの推論を行う。こちらも処理結果は`result.png`という画像ファイルに保存される。

```
python face_detector.py facenet models/facenet.npz --img data/face.png 
```


<div align="center">
<img src="data/face.png" width="300">
&nbsp;
<img src="data/face_result.png" width="300">
</div>


同様に、以下のコマンドで手のランドマークの推論を行う。こちらも処理結果は`result.png`という画像ファイルに保存される。

```
python hand_detector.py handnet models/handnet.npz --img data/hand.png 
```

<div align="center">
<img src="data/hand.png" width="300">
&nbsp;
<img src="data/hand_result.png" width="300">
</div>


同様に、以下のコマンドでポーズ、顔、及び両手の全てのランドマークの推論を行う。こちらも処理結果は`result.png`という画像ファイルに保存される。

```
python demo.py --img data/dinner.png
```

<div align="center">
<img src="data/dinner.png" width="340">
&nbsp;
<img src="data/dinner_result.png" width="340">
</div>


ウェブカメラをお使いの場合は、以下のコマンドで、カメラの画像を入力として推論を行うリアルタイムデモを実行する事ができる。`q` キーで終了する。

<b>リアルタイムポーズ推定：</b>

```
python camera_pose_demo.py
```

<b>リアルタイム顔推定：</b>

```
python camera_face_demo.py
```


## 訓練手順
COCO 2017を使ったフルスクラッチでの訓練手順

### COCO 2017のデータをダウンロード

```
bash getData.sh
```
すでにデータセットをダウンロード済みの場合、`entity.py`中のcoco_dirにCOCOデータセットのパスを記入する。

### cocoapiのセットアップ

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../../
```


### data generatorの確認
以下のコマンドで、generatorを使ってランダム生成された訓練用画像を確認できる。切り出し画像に対して正解のPAFとHeatmap及びmaskを被せた状態で表示される。

```
python coco_data_loader.py
```

### COCOデータセットで訓練

1000イテレーションごとに、その時点の重みパラメータが `model_iter_1000` というような重みファイルに保存される。

```
python train_coco_pose_estimation.py --gpu 0
```

### 訓練したモデルで推論

自前で訓練したモデルを使って推論処理を行う場合は、同じように以下のコマンドで訓練済み重みパラメータファイルと画像を指定すれば良い。処理結果は`result.png`という画像ファイルに保存される。

```
python pose_detector.py posenet model_iter_1000 --img data/person.png 
```




## 関連リポジトリ
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).


## Citation
Please cite the paper in your publications if it helps your research:    

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
	  
