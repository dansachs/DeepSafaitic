# Setting Up GitHub Repository

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `DeepSafaitic`
3. Description: "Neural Epigraphy for Ancient Desert Inscriptions - Proof of concept training pipeline for reading Safaitic rock inscriptions"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Add Remote and Push

After creating the repository, run these commands (replace `YOUR_USERNAME` with your GitHub username):

```bash
# Add the remote
git remote add origin https://github.com/YOUR_USERNAME/DeepSafaitic.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

Check your repository at: `https://github.com/YOUR_USERNAME/DeepSafaitic`

## Optional: Add Topics/Tags

On GitHub, click the gear icon next to "About" and add topics:
- `epigraphy`
- `ancient-scripts`
- `safaitic`
- `computer-vision`
- `deep-learning`
- `siamese-networks`
- `archaeology`
- `neural-networks`
- `pytorch`
- `ocr`

## Repository Description

Update the repository description to:
"Proof of concept for building a training pipeline that will allow models to read and interpret Safaitic rock inscriptions using Siamese neural networks and computer vision."
