# BLAST_Ripper-Meta: Parallel BLAST Processing Pipeline with Taxonomic Analysis

## Overview

**BLAST_Ripper-Meta** is an integrated pipeline designed to perform parallel BLAST searches on multiple FASTA files, process the results by adding taxonomic information, and generate visualizations of taxonomic distributions.

This script is optimized for efficient analysis of large sequencing datasets and offers the following key features:

- **Parallel BLAST Processing**: Utilizes multiprocessing to process multiple FASTA files in parallel, enhancing speed and efficiency.
- **Comprehensive Taxonomic Integration**: Incorporates detailed taxonomic information by parsing local taxonomy databases.
- **Advanced Result Parsing and Visualization**: Parses BLAST results to generate various taxonomic distribution visualizations.
- **Performance Monitoring and Logging**: Monitors memory and CPU usage throughout the process and records detailed logs.
- **Memory Usage Optimization**: Implements memory-efficient algorithms for handling large-scale data processing.

---

## Key Features Summary

1. **Parallel BLAST Processing**: Splits input FASTA files into multiple chunks and performs BLAST searches in parallel.
2. **Taxonomic Information Augmentation**: Parses local taxonomy databases (`names.dmp`, `nodes.dmp`) to enrich sequences with taxonomic lineage information.
3. **Result Parsing and Filtering**: Parses BLAST results and filters matches based on user-defined coverage and identity thresholds.
4. **Visualization Generation**: Creates various visualizations of taxonomic distributions (sunburst plots, heatmaps, etc.).
5. **Report Generation**: Produces detailed HTML and text reports summarizing the analysis results.
6. **Performance Monitoring**: Monitors memory and CPU usage during processing and logs performance metrics.

---

## Installation and Dependencies

### Required Dependencies

The following Python packages are required:

- Python 3.6 or higher
- Biopython
- Matplotlib
- NumPy
- Pandas
- Seaborn
- tqdm
- NetworkX
- psutil

### Installation Instructions

Using `conda`:

```bash
conda create -n blast_ripper_env python=3.8
conda activate blast_ripper_env
conda install -c conda-forge biopython matplotlib numpy pandas seaborn tqdm networkx psutil
```

Using `pip`:

```bash
pip install biopython matplotlib numpy pandas seaborn tqdm networkx psutil
```

---

## Usage

### Running the Script

```bash
python blast_ripper_meta.py -in <input_folder> -out <output_folder> -db <BLAST_database_path> -taxdb <taxonomy_database_path> [options]
```

### Required Arguments

- `-in`, `--input_folder`: Path to the input folder containing FASTA files.
- `-out`, `--output_folder`: Path to the output folder where results will be stored.
- `-db`, `--db`: Path to the BLAST database (e.g., the `nt` database).
- `-taxdb`, `--taxdb_dir`: Path to the taxonomy database containing `names.dmp` and `nodes.dmp` files.

### Optional Options

- `-t`, `--threads`: Number of threads per BLAST process (default: 4).
- `-n`, `--num_chunks`: Number of chunks to split each input file into (default: 8).
- `-shd`, `--use-ramdisk`: Use `/dev/shm` as temporary storage to reduce disk I/O.
- `-taxids`: List of taxids to filter results (optional).
- `-qcov`, `--qcov_threshold`: Query coverage threshold for considering a match (default: 80%).
- `-identity`, `--identity_threshold`: Identity threshold (default: 90%).
- `-h`, `--help`: Show help message and exit.

### Example Command

```bash
python blast_ripper_meta.py -in input_folder -out output_folder -db /path/to/nt -taxdb /path/to/taxonomy -t 8 -n 16 -qcov 85 -identity 95
```

---

## Detailed Explanation

### 1. Input File Processing

The script scans the input folder for all files ending with `.fasta` or `.fa` and processes each one. Each file is split into the specified number of chunks and processed in parallel using multiprocessing.

### 2. BLAST Search

A BLAST search is performed on each chunk, and the results are saved in TSV format. The `-taxids` option allows you to filter results by specific taxids.

### 3. Parsing BLAST Results

BLAST results are parsed, and matches meeting the specified query coverage and identity thresholds are selected. The best match for each sequence is retained.

### 4. Adding Taxonomic Information

The script parses the local taxonomy database (`names.dmp` and `nodes.dmp`) to build detailed taxonomic information for each taxid. This includes species, genus, family, order, and other lineage information, which are added to the sequence matches.

### 5. Saving Results

For each input file, the final results are saved in TSV format, including sequence ID, match status, and taxonomic information.

### 6. Generating Visualizations

Based on the analyzed data, various visualizations are generated:

- **Sunburst Plot**: Visualizes the distribution according to taxonomic hierarchy.
- **Heatmap**: Compares diversity metrics across samples.
- **Network Diagram**: Illustrates taxonomic relationships.
- **Diversity Metrics Bar Graph**: Visualizes species richness, Shannon index, Simpson index, etc.

### 7. Report Generation

Detailed HTML and text reports summarizing the analysis and providing comprehensive statistics are generated.

---

## Output File Structure

- `<output_folder>/`: Output directory
  - `<input_file_name>/`: Subdirectory for each input file
    - `*_combined.tsv`: Combined BLAST results
    - `*_final_results.tsv`: Final results file
  - `visualizations/`: Directory containing visualization images
    - `sunburst_plot.png`
    - `diversity_heatmap.png`
    - `taxonomy_network.png`
    - Other visualization images
  - `diversity_metrics.json`: File containing diversity metrics
  - `summary_report.txt`: Summary report
  - `detailed_report.tsv`: Detailed report
  - `final_report.html`: Final HTML report
  - `analysis_report.html`: Report index page
  - `*.log`: Log files

---

## Logging and Monitoring

The script monitors memory and CPU usage throughout the process and records detailed logs. Log files are saved in the output directory with a timestamp.

---

## Notes and Cautions

- **BLAST Database**: Ensure that the latest BLAST database is installed locally before running the script.
- **Taxonomy Database**: A taxonomy database containing `names.dmp` and `nodes.dmp` is required.
- **System Resources**: Due to the processing of large datasets, it's recommended to run the script on a system with sufficient memory and CPU cores.
- **Using RAM Disk**: The `-shd` option uses `/dev/shm` as temporary storage to reduce disk I/O. Ensure that your system has enough free memory to accommodate this.

---

## Troubleshooting

- **Dependency Errors**: Ensure all required Python packages are installed.
- **Memory Issues**: If you encounter memory errors during processing, consider reducing the number of chunks using the `-n` option or avoid using the RAM disk.
- **BLAST Errors**: If errors occur during BLAST execution, verify the BLAST database path and check for appropriate permissions.

---

## Contact

For questions or suggestions regarding the script, please contact the maintainer.

---

## License

This script is distributed under the MIT License.

---

**Thank you for using BLAST_Ripper-Meta!**
