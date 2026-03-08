# GitHub SSH Setup & Deployment Guide

This guide provides step-by-step instructions for setting up GitHub authentication via SSH and managing deployments on new instances (e.g., Vast.ai, Lightning.ai).

---
## FIRST TRY WITH gh auth login
select ssh

## 1. SSH Key Setup

### Step 1: Check for Existing Keys
Run this to see if you already have an SSH key:
```bash
ls -al ~/.ssh/id_ed25519.pub
```
*If you see a file, skip to Step 3. If you get "No such file", proceed to Step 2.*

### Step 2: Generate a New Key
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
- Press **Enter** to save in the default location.
- Press **Enter** (twice) for **no passphrase** (recommended for automated/cloud instances).

### Step 3: Add Key to GitHub
1. Copy the public key text:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
2. Go to [GitHub Settings > SSH and GPG keys](https://github.com/settings/keys).
3. Click **New SSH key**, give it a name (e.g., "Vast.ai Instance"), and paste the text.

### Step 4: Test Connection
```bash
ssh -T git@github.com
```
*Type `yes` if prompted. You should see "Hi [username]! You've successfully authenticated".*

---

## 2. Repository Deployment

### Clone a New Instance
```bash
git clone git@github.com:Sonicprint/LTX-2.git
cd LTX-2
```

### Switch an Existing Instance from HTTPS to SSH
If you already have the repo but it keeps asking for a password:
```bash
git remote set-url origin git@github.com:Sonicprint/LTX-2.git
```

---

## 3. Typical Deployment Workflow

### Pull Latest Changes
Run this whenever you want to get the latest fixes or features:
```bash
git pull origin main
```

### Push Your Local Changes
1. **Stage changes**: `git add .`
2. **Commit**: `git commit -m "Brief description of changes"`
3. **Push**: `git push origin main`

### Quick Fix for Authentication Errors
If you get a connection error, verify your SSH agent is running:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

## 4. LTX-2 Specific Setup
After cloning, don't forget to run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```
