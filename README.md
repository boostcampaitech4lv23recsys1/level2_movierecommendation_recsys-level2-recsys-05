# Movie Recommendation
![image](https://user-images.githubusercontent.com/86918832/215697852-10f249ed-288b-4fc0-8157-8cfb9fe41fa8.png)
본 프로젝트의 목표는 MovieLens 데이터셋을 이용하여 추천 모델을 구축하는 것입니다. 사용자의 영화 시청 이력을 바탕으로 사용자가 다음에 시청할 영화 혹은 영화와 영화 중간에 들어갈 영화를 예측합니다. 평가 지표는 Recall@10 입니다.

## 팀원 소개
|이름|역할|
|----|---|
|[강민수](https://github.com/minsu0216)|EDA, Model experiments (BPR, WideDeep, FM, FFM, DeepFM, DCN, SASRecF)|
|[김진명](https://github.com/tobe-honest)|EDA, Feature Engineering, Model experiments (BERT4Rec, SASRec, RECVAE, S3Rec)|
|[박경태](https://github.com/GT0122)|EDA, Feature Engineering, Model experiments (MultiVAE, LightGCN)|
|[박용욱](https://github.com/oceanofglitta)|EDA, Model experiments (EASE, GRU4Rec, RecVAE, S3Rec)|

## 활용 장비 및 도구
- 서버: V100 GPU 서버
- 개발 IDE: Jupyter Notebook, VS Code
- 협업 Tool: Notion, Slack, Zoom, Github

## 데이터 구조
```python
train
    ├── Ml_item2attributes.json      # category -> index 
    ├── directors.tsv                # director by movie data
    ├── genres.tsv                   # genre by movie data
    ├── titles.tsv                   # title by movie data
    ├── train_ratings.csv            # interaction (user, item, timestamp)
    ├── writers.tsv                  # writer by movie data
    └── years.tsv                    # year by movie data
```

## 폴더 구조
```bash
level2_movierecommendation_recsys-level2-recsys-05
├── code                    # 대회 baseline 코드
│   ├── base
│   ├── data_loader
│   ├── datasets
│   ├── logger
│   ├── model
│   ├── trainer
│   └── utils
├── eda                     # EDA 코드
└── feature_engineering     # Feature Engineering 코드
```

## 수행 결과
- EASE 단일 모델
- Recall@10 : 0.1600(public, 12th) -> 0.1600(private, 12th)