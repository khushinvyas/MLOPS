# Production Deployment Guide

## Overview
This guide explains how to deploy your MLOps energy prediction app to EC2 using Docker and GitHub Actions CI/CD.

## Architecture

```
GitHub Push → GitHub Actions → Docker Hub → EC2 (SSM) → Container Running
```

1. **Code Push**: You push code to `master` branch
2. **CI/CD Pipeline**: GitHub Actions automatically:
   - Builds Docker image
   - Pushes to Docker Hub
   - Deploys to EC2 via AWS SSM
3. **EC2 Deployment**: Container automatically:
   - Pulls latest image
   - Downloads models from S3
   - Serves app on port 5000

## Prerequisites Checklist

### ✅ EC2 Instance Setup
- [x] Ubuntu instance with Docker installed
- [x] Security Group allows inbound port 5000 (or 80 if using Nginx)
- [x] IAM Role attached with S3 read permissions for `s3://khushin/*`
- [x] SSM Agent installed and running (usually pre-installed on Amazon Linux/Ubuntu)

### ✅ Docker Hub Setup
- [x] Docker Hub account created
- [x] Repository created: `YOUR_USERNAME/energy-app`
- [x] Access token generated (Settings → Security → New Access Token)

### ✅ GitHub Secrets Configured
Go to: `Settings → Secrets and variables → Actions`

Required secrets:
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token
- `AWS_ACCESS_KEY_ID`: AWS IAM user access key (with SSM permissions)
- `AWS_SECRET_ACCESS_KEY`: AWS IAM user secret key
- `AWS_REGION`: Your AWS region (e.g., `eu-north-1`)
- `AWS_EC2_INSTANCE_ID`: Your EC2 instance ID (e.g., `i-0123456789abcdef0`)

### ✅ S3 Models
- [x] Models uploaded to `s3://khushin/mlops-final/models/`:
  - `xgb_model.pkl`
  - `rf_model.pkl`
  - `lgbm_model.pkl`

## Files Created

### 1. `.github/workflows/docker-deploy.yaml`
- **Purpose**: Automated CI/CD pipeline
- **Triggers**: Push to `master` branch (only on code changes)
- **Actions**:
  1. Builds Docker image
  2. Pushes to Docker Hub
  3. Deploys to EC2 via SSM

### 2. `Dockerfile`
- **Purpose**: Containerizes Flask app
- **Base**: Python 3.11-slim
- **Exposes**: Port 5000
- **Command**: Gunicorn serves Flask app

### 3. `.dockerignore`
- **Purpose**: Excludes unnecessary files from Docker build
- **Excludes**: `.dvc/`, `data/`, `models/`, etc.

## How It Works

### First Deployment

1. **Push code to GitHub**:
   ```bash
   git add .github/workflows/docker-deploy.yaml Dockerfile .dockerignore
   git commit -m "Add Docker deployment workflow"
   git push origin master
   ```

2. **GitHub Actions runs automatically**:
   - Check Actions tab in GitHub
   - Watch the workflow execute

3. **EC2 receives deployment**:
   - Container starts automatically
   - Models download from S3 on first startup
   - App available at `http://EC2_PUBLIC_IP:5000`

### Subsequent Deployments

- Simply push code to `master`
- GitHub Actions handles everything automatically
- Zero-downtime deployment (old container stops, new one starts)

## Model Management

### Current Setup
- Models stored in S3: `s3://khushin/mlops-final/models/`
- Container downloads models on startup (if not in volume)
- Models persist in Docker volume `energy_models` between restarts

### Updating Models
1. Train new models locally
2. Upload to S3:
   ```bash
   aws s3 cp models/xgb_model.pkl s3://khushin/mlops-final/models/xgb_model.pkl
   aws s3 cp models/rf_model.pkl s3://khushin/mlops-final/models/rf_model.pkl
   aws s3 cp models/lgbm_model.pkl s3://khushin/mlops-final/models/lgbm_model.pkl
   ```
3. Restart container on EC2 (or wait for next deployment)

### Alternative: Include Models in Image
If you want models baked into the image (not recommended for size):
- Modify Dockerfile to copy models
- Trade-off: Larger image, but faster startup

## Troubleshooting

### Deployment Fails
1. Check GitHub Actions logs
2. Verify all secrets are set correctly
3. Check EC2 instance status and SSM connectivity:
   ```bash
   aws ssm describe-instance-information --region eu-north-1
   ```

### Container Won't Start
1. SSH into EC2 (or use SSM Session Manager)
2. Check Docker logs:
   ```bash
   docker logs energy-app
   ```
3. Verify S3 access:
   ```bash
   docker exec energy-app aws s3 ls s3://khushin/mlops-final/models/
   ```

### Models Not Loading
1. Verify S3 bucket permissions on EC2 IAM role
2. Check environment variables in container:
   ```bash
   docker exec energy-app env | grep MODEL
   ```
3. Manually test S3 download:
   ```bash
   docker exec energy-app aws s3 cp s3://khushin/mlops-final/models/xgb_model.pkl /tmp/test.pkl
   ```

## Security Best Practices

1. **Use IAM Roles** (not access keys in container):
   - EC2 instance should have IAM role with S3 read permissions
   - Container automatically uses instance role

2. **Rotate Secrets**:
   - Regularly rotate Docker Hub tokens
   - Rotate AWS access keys

3. **Limit SSM Access**:
   - IAM user for GitHub Actions should have minimal permissions:
     - `ssm:SendCommand`
     - `ssm:GetCommandInvocation`
     - `ssm:DescribeInstanceInformation`

4. **Use HTTPS**:
   - Add Nginx reverse proxy with SSL certificate
   - Use Let's Encrypt for free SSL

## Monitoring

### Check Container Status
```bash
# On EC2
docker ps | grep energy-app
docker logs energy-app --tail 50
```

### Health Check Endpoint
Add to `app.py`:
```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'models_loaded': len(LOADED_MODELS)}
```

Then monitor: `http://EC2_PUBLIC_IP:5000/health`

## Next Steps

1. **Add Nginx** (optional but recommended):
   - Reverse proxy on port 80
   - SSL certificate
   - Better security headers

2. **Set up Monitoring**:
   - CloudWatch for logs
   - Health check alerts
   - Model performance tracking

3. **Add Rollback**:
   - Keep previous image tag
   - Quick rollback if deployment fails

4. **Automate Model Updates**:
   - Add step to workflow to upload models after training
   - Trigger model refresh automatically

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Check EC2 Docker logs
3. Verify all prerequisites are met
4. Review AWS SSM command execution logs

