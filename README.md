

#### **Section A: COMPLETE ALL REQUIRED STEPS IN ORDER TO REPRODUCE ALL RESULTS**



### **Step 1 --- Google Drive Preparation**
*   Create a folder named `pe_experiment` in your root Google Drive directory.
*   **Final Path on Drive:** `/My Drive/pe_experiment/`
*   **Note:** In Colab, the full path will be: `/content/drive/MyDrive/pe_experiment/`


---

### **Step 2 --- Data Setup & Structure**

⚠️ **IMPORTANT:** The folder structure must be identical to the diagram below (Section B). All script paths are hardcoded.

1.  **From GitHub:** Download the repository and copy the following files into the root folder `/pe_experiment/`:
    *   The `table1_scripts` folder (including all 14 scripts).
    *   Python scripts: `00_setup_imagenet.py` and `full_scale_experiment.py`.
    *   Text files: `imagenet100_classes.txt` and `val_labels.txt`.
    *   **The Colab notebook: `reproduce_paper_results.ipynb`.**

2.  **ImageNet Dataset Acquisition:** Create a folder named `imagenet` within `/pe_experiment/`. 
    *   ImageNet requires registration. Go to image-net.org and register with an academic email. Once approved, you will receive an email with a unique download link. Use it to download the ILSVRC2012_img_val.tar archive (~6.3 GB).
    *   Place the archive inside /imagenet/. **Do NOT extract the archive.** The Colab script will handle the extraction automatically.


3.  **Model Results (7.6 GB Total):** 
    *   Open the [ImageNet trained models](https://drive.google.com/drive/folders/1gPwVSE0qctWVeaGwCv3eGQdQR4IK6Xds?usp=sharing) and [CIFAR100 trained models](https://drive.google.com/drive/folders/16pEAbdH4aRpw-3s2vm4TbMey1GPQn2FQ?usp=sharing) links.
    **Note on Access:** You have **Viewer** access. To use these files, you must **Select all items** inside the link, right-click, and choose **"Make a copy"**.
    *   Google will create copies in your root "My Drive". **Move them** into their respective folders: `/results/` and `/results_cifar100/`. Each model folder must contain `best_model.pth` and `training_history.json`.
    *   ⚠️ Important: Ensure all folders and files are named exactly as shown (remove any "Copy of" prefixes added by Google Drive).
    *   ⚠️ Note: The folder /pe_experiment/results/ must also contain the two shared analysis files: adversarial_pe_results.json and analysis_data.json (see the diagram in Section B).
    *   ⚠️ Note: The folder /pe_experiment/results_cifar100/ must also contain the shared file: adversarial_pe_results_cifar100.json (see the diagram in Section B).


---

### **Step 3 --- Launch in Colab**
*   Open the `reproduce_paper_results.ipynb` notebook from your Google Drive.
*   **Runtime:** Go to **Runtime > Change runtime type** and select **GPU (H100 or A100)**.
*   **Mount & Setup:** Run the **first cell (Cell 1)**. 
    *   When the popup appears, click **"Connect to Google Drive"** to grant access.
    *   **Note:** This cell will automatically mount your Drive and **copy all necessary scripts from your Drive into the Colab local environment (SSD)**.


---

### **Step 4 --- Code Execution**
*   **Starting from Cell 2:** Run the remaining cells one by one in the given order, from top to bottom.
*   **Sequential Execution:** Ensure each cell finishes completely before starting the next one to maintain the correct data flow.
*   **Automatic Saving:** The scripts will automatically create the `figures/`and `table1/` folders within `/pe_experiment/results/` and **save/store all generated outputs (results) directly into them.**

### **Verification:**
Once execution is complete, you can compare your generated outputs in /results/figures/ and /results/table1/ with the original study results provided in the original_paper_results/ folder for consistency.



#### **Section B: STRUCTURE BEFORE START OF REPRODUCTION**
```text
/My Drive/pe_experiment/
├── table1_scripts/             # (14 scripts)
├── imagenet/
│   └── ILSVRC2012_img_val.tar  # (Keep archived!)
├── results/                    # ImageNet100 results
│   ├── alibi_seed42/
│   ├── alibi_seed123/
│   ├── alibi_seed456/
│   ├── learned_seed42/
│   ├── learned_seed123/
│   ├── learned_seed456/
│   ├── rope_seed42/
│   ├── rope_seed123/
│   ├── rope_seed456/
│   ├── sinusoidal42/
│   ├── sinusoidal123/
│   ├── sinusoidal456/
│   ├── adversarial_pe_results.json
│   └── analysis_data.json
├── results_cifar100/           # CIFAR100 results
│   ├── alibi_seed42/
│   ├── alibi_seed123/
│   ├── alibi_seed456/
│   ├── learned_seed42/
│   ├── learned_seed123/
│   ├── learned_seed456/
│   ├── rope_seed42/
│   ├── rope_seed123/
│   ├── rope_seed456/
│   ├── sinusoidal42/
│   ├── sinusoidal123/
│   ├── sinusoidal456/
│   └── adversarial_pe_results_cifar100.json
├── imagenet100_classes.txt
├── val_labels.txt
├── 00_setup_imagenet.py
├── full_scale_experiment.py
└── reproduce_paper_results.ipynb
