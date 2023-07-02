# Readme for Homework 4

## Docker Build
```bash
docker build -t homework_week4 .
```


## Docker Run
I had to run Docker Build like:
- I'm using Mac M1 2021 version
```bash
docker run --platform linux/amd64 homework_week4 yellow 2021 4
```

## Boto3 Credentials
- Kept a .env inside of my homework/ root dir, where I stored the details like:
```env
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY_ID=...
```
