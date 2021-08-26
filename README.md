# multi-label-classification
데이콘의 제2회 컴퓨터 비전 학습 경진대회를 연습해보았다.

## Dataset
```
https://dacon.io/competitions/official/235697/overview/description
```

## Docker 실행
※ 만약 Jupyter Lab 환경에서 실행시키고 싶으신분들은 DockerFile에서 주석을 제거해 주시면 됩니다.
1. Repository의 폴더로 가서 make run을 실행한다.
2. 브라우저에서 jupyter lab(127.0.0.1:9999)에 접속한다. 
3. 생성된 .secret 파일에서 token 값을 읽는다.
4. 읽은 token으로 jupyter lab을 인증한다.

```shell
# 서버 실행
make run

# 서버 다운
make stop
```


## Reference
[Code](https://dacon.io/competitions/official/235697/codeshare/2440?page=1&dtype=recent)