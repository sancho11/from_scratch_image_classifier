# get\_dataset.md

This document explains how to download and prepare the dataset required for running this project, specifically the **From-Scratch Image Classification** data from the Kaggle competition.

## Prerequisites

1. **Kaggle account**: You need a Kaggle account to access the competition data. If you don't have one, register at [https://www.kaggle.com](https://www.kaggle.com).
2. **Kaggle API token**: Generate an API token:

   * Go to your Kaggle account settings ([https://www.kaggle.com/\`](https://www.kaggle.com/`)<your-username>\`/account).
   * Scroll to **API** and click **Create New API Token**.
   * A file named `kaggle.json` will be downloaded.
3. **Kaggle CLI**: Install the Kaggle command-line interface:

   ```bash
   pip install kaggle
   ```
4. **Place the token**: Move `kaggle.json` to the configuration directory:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Downloading the Dataset

### Using Kaggle CLI

1. **Download competition data**:

   ```bash
   kaggle competitions download \
     -c open-cv-py-torch-project-1-classification-round-2 \
     -p ./
   ```

   This will download a zip file (e.g., `open-cv-py-torch-project-1-classification-round-2.zip`) into the project's root folder.
2. **Unzip the files**:

   ```bash
   unzip open-cv-py-torch-project-1-classification-round-2.zip
   ```
3. **Verify file structure**. After extraction, you should see:

   ```text
   dataset/
   ├── Train/        # training images
   ├── Test/         # testing images
   └── Valid/        # validation images
   ```

### Alternative Download (Without Kaggle CLI)

1. Open your web browser and navigate to the competition page:
   [https://www.kaggle.com/competitions/open-cv-py-torch-project-1-classification-round-2](https://www.kaggle.com/competitions/open-cv-py-torch-project-1-classification-round-2)
2. Log in with your Kaggle account and accept the competition rules.
3. On the **Data** tab, click **Download All** to download the ZIP file.
4. Place the downloaded ZIP into project's root directory:

   ```bash
   mv ~/Downloads/open-cv-py-torch-project-1-classification-round-2.zip ./
   ```
5. Extract the archive:

   ```bash
   unzip open-cv-py-torch-project-1-classification-round-2.zip
   ```
6. Confirm you have:

   ```text
   dataset/
   ├── Train/
   ├── Test/
   └── Valid/
   ```
