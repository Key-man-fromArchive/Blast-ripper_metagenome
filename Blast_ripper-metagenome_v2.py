import argparse
import subprocess
import tempfile
import os
import shlex
from multiprocessing import Pool, cpu_count
from Bio import SeqIO, Entrez
import shutil
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import json
from Bio import Phylo
from io import StringIO
import logging
from tqdm import tqdm
import psutil
import time
import networkx as nx
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(output_dir):
    """Set up logging configuration with rotation"""
    log_file = os.path.join(output_dir, f'blast_ripper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 로그 회전 설정 추가
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    
    return logger

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def monitor_cpu():
    """Monitor current CPU usage"""
    return psutil.cpu_percent(interval=1)

class PerformanceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting process. Initial memory usage: {monitor_memory():.2f} MB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.info(f"Process completed in {duration:.2f} seconds")
        self.logger.info(f"Final memory usage: {monitor_memory():.2f} MB")
        
class HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        title = (
            "╔╗ ┬  ┌─┐┌─┐┌┬┐\n"
            "╠╩╗│  ├─┤└─┐ │ \n"
            "╚═╝┴─┘┴ ┴└─┘ ┴ \n"
            "╦═╗┬┌─┐┌─┐┌─┐┬─┐\n"
            "╠╦╝│├─┘├─┘├┤ ├┬┘\n"
            "╩╚═┴┴  ┴  └─┘┴└─\n"
            "╔╦╗┌─┐┌┬┐┌─┐\n"
            " ║║║├┤  │ ├─┤\n"
            "═╩╝└─┘ ┴ ┴ ┴"
        )
        description = """
BLAST_Ripper-Meta: A Parallel BLAST Processing Pipeline with Taxonomy Analysis

Overview:
This script performs BLAST searches on multiple FASTA files in parallel, processes the results,
adds taxonomic information, and generates visualizations of the taxonomic distribution.

Key Features:
1. Parallel BLAST processing with CPU optimization
2. Comprehensive taxonomic information retrieval
3. Advanced result parsing and visualization
4. Performance monitoring and logging
5. Memory usage optimization
"""
        print(title)
        print(description)
        print(parser.format_help())
        parser.exit()

def run_blast(chunk, blast_db, out_file, use_ramdisk, num_threads, taxids, logger):
    with PerformanceMonitor(logger):
        logger.info(f"Running BLAST on chunk: {chunk}")
        if use_ramdisk:
            temp_chunk = os.path.join('/dev/shm', os.path.basename(chunk))
            temp_out = os.path.join('/dev/shm', os.path.basename(out_file))
            shutil.copy(chunk, temp_chunk)
        else:
            temp_chunk = chunk
            temp_out = out_file

        taxid_option = f"-taxids {','.join(taxids)}" if taxids else ""
        
        cmd = (f"blastn -query {temp_chunk} -db {blast_db} "
            f"-outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids qcovs qlen' "
            f"-max_target_seqs 5 -num_threads {num_threads} "
            f"{taxid_option} -out {temp_out}")
        
        try:
            result = subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True, timeout=3600)
            if "The -taxids command line option requires additional data files" in result.stderr:
                logger.warning("Taxid filtering is not available. Running BLAST without taxid filter.")
                cmd_without_taxid = cmd.replace(taxid_option, "")
                subprocess.run(shlex.split(cmd_without_taxid), check=True, timeout=3600)
        except subprocess.TimeoutExpired:
            logger.error(f"BLAST process timed out for chunk: {chunk}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running BLAST: {e}")
            logger.error(f"BLAST stderr: {e.stderr}")
            return None

        if use_ramdisk:
            shutil.move(temp_out, out_file)
            os.remove(temp_chunk)
        logger.info(f"Completed BLAST for chunk: {chunk}")
        return out_file

def parse_local_taxonomy(taxdb_dir, logger):
    """Parse local taxonomy database with improved error handling and logging"""
    with PerformanceMonitor(logger):
        taxonomy_dict = {}
        logger.info(f"Starting taxonomy database parsing from {taxdb_dir}")
        
        # Parse names.dmp
        names_file = os.path.join(taxdb_dir, 'names.dmp')
        logger.info(f"Parsing names from: {names_file}")
        scientific_names = {}
        
        try:
            with open(names_file, 'r') as f:
                for line_num, line in enumerate(tqdm(f, desc="Parsing names.dmp"), 1):
                    fields = line.split('|')
                    if len(fields) > 3 and fields[3].strip() == "scientific name":
                        taxid = fields[0].strip()
                        name = fields[1].strip()
                        scientific_names[taxid] = name
                    
                    if line_num % 100000 == 0:
                        logger.debug(f"Processed {line_num} lines from names.dmp")
        except Exception as e:
            logger.error(f"Error parsing names.dmp: {str(e)}")
            raise

        logger.info(f"Total scientific names parsed: {len(scientific_names)}")

        # Parse nodes.dmp
        nodes_file = os.path.join(taxdb_dir, 'nodes.dmp')
        logger.info(f"Parsing nodes from: {nodes_file}")
        
        try:
            with open(nodes_file, 'r') as f:
                for line_num, line in enumerate(tqdm(f, desc="Parsing nodes.dmp"), 1):
                    fields = line.split('|')
                    if len(fields) > 2:
                        taxid = fields[0].strip()
                        parent_taxid = fields[1].strip()
                        rank = fields[2].strip()
                        
                        if taxid in scientific_names:
                            name = scientific_names[taxid]
                            taxonomy_dict[taxid] = {
                                'name': name,
                                'rank': rank,
                                'parent_taxid': parent_taxid,
                                'lineage': {}
                            }
                    
                    if line_num % 100000 == 0:
                        logger.debug(f"Processed {line_num} lines from nodes.dmp")

            # Build lineage information
            logger.info("Building lineage information...")
            for taxid in tqdm(taxonomy_dict.keys(), desc="Building lineages"):
                current_taxid = taxid
                visited = set()  # Prevent infinite loops
                
                while current_taxid != '1' and current_taxid in taxonomy_dict and current_taxid not in visited:
                    visited.add(current_taxid)
                    parent_taxid = taxonomy_dict[current_taxid]['parent_taxid']
                    parent_info = taxonomy_dict.get(parent_taxid, {})
                    parent_rank = parent_info.get('rank', '')
                    parent_name = parent_info.get('name', '')

                    if parent_rank in ['species', 'genus', 'family', 'order']:
                        taxonomy_dict[taxid]['lineage'][parent_rank] = parent_name

                    current_taxid = parent_taxid

        except Exception as e:
            logger.error(f"Error parsing nodes.dmp: {str(e)}")
            raise

        logger.info(f"Total taxonomy entries: {len(taxonomy_dict)}")
        return taxonomy_dict

def process_file(file_path, output_dir, db, num_chunks, threads, use_ramdisk, taxids, taxonomy_dict, qcov_threshold, logger):
    """Process a single FASTA file with improved error handling and monitoring"""
    with PerformanceMonitor(logger):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        output_subdir = os.path.join(output_dir, base_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Split file into chunks
        logger.info(f"Splitting {file_name} into {num_chunks} chunks")
        chunks = chunk_file(file_path, num_chunks, logger)
        
        # Run BLAST on chunks
        chunk_outputs = []
        max_processes = min(num_chunks, cpu_count())
        logger.info(f"Using {max_processes} processes for parallel BLAST")
        
        with Pool(processes=max_processes) as p:
            chunk_outputs = list(tqdm(
                p.starmap(run_blast, [
                    (chunk, db, os.path.join(output_subdir, f"{base_name}_chunk_{i}.tsv"),
                     use_ramdisk, threads, taxids, logger)
                    for i, chunk in enumerate(chunks)
                ]),
                total=len(chunks),
                desc="Running BLAST"
            ))

        # Combine outputs
        combined_output = os.path.join(output_subdir, f"{base_name}_combined.tsv")
        logger.info("Combining BLAST outputs")
        with open(combined_output, 'w') as outfile:
            for chunk_output in chunk_outputs:
                if chunk_output and os.path.exists(chunk_output):
                    with open(chunk_output) as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_output)
                else:
                    logger.warning(f"BLAST output missing for a chunk in {file_path}")

        # Process results
        logger.info("Processing BLAST results")
        matched_sequences = parse_blast_results(combined_output, qcov_threshold, logger)

        # Generate final output
        final_output = os.path.join(output_subdir, f"{base_name}_final_results.tsv")
        generate_final_output(file_path, matched_sequences, taxonomy_dict, final_output, logger)

        # Clean up temporary files
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)
        
        # Collect taxids
        file_taxids = set()
        for matches in matched_sequences.values():
            for match in matches:
                taxid = match[2]  # taxid from match tuple
                file_taxids.add(taxid)

        return file_taxids, matched_sequences

def get_taxonomy_info(taxid, taxonomy_dict):
    """Get formatted taxonomy information for a given taxid"""
    if taxid in taxonomy_dict:
        info = taxonomy_dict[taxid]
        return {
            'species': info['name'] if info['rank'] == 'species' else 'N/A',
            'genus': info['lineage'].get('genus', 'N/A'),
            'family': info['lineage'].get('family', 'N/A'),
            'order': info['lineage'].get('order', 'N/A')
        }
    return {'species': 'N/A', 'genus': 'N/A', 'family': 'N/A', 'order': 'N/A'}

def generate_final_output(input_file, matched_sequences, taxonomy_dict, output_file, logger):
    """Generate final output file with matches and taxonomy information"""
    logger.info("Generating final output file")
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow([
            'Sequence_ID', 'Sequence', 'Match_Status', 'Subject_ID',
            'Percent_Identity', 'Query_Coverage', 'Alignment_Length',
            'Query_Length', 'Taxid', 'Scientific_Name', 'Genus', 'Family', 'Order'
        ])
        
        for record in tqdm(SeqIO.parse(input_file, "fasta"), desc="Writing results"):
            sequence_id = record.id
            query_length = len(record.seq)
            
            if sequence_id in matched_sequences:
                for match in matched_sequences[sequence_id]:
                    subject_id, percent_identity, taxid, query_coverage, alignment_length, _ = match
                    taxonomy = get_taxonomy_info(taxid, taxonomy_dict)
                    
                    writer.writerow([
                        sequence_id, str(record.seq), 'Matched', subject_id,
                        percent_identity, f"{query_coverage:.2f}", alignment_length,
                        query_length, taxid, taxonomy['species'], taxonomy['genus'],
                        taxonomy['family'], taxonomy['order']
                    ])
            else:
                writer.writerow([
                    sequence_id, str(record.seq), 'Unmatched', 'N/A', 'N/A',
                    'N/A', 'N/A', query_length, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])

class TaxonomyAnalyzer:
    def __init__(self, taxonomy_dict, logger):
        self.taxonomy_dict = taxonomy_dict
        self.logger = logger
        self.rank_order = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
    def build_lineage_tree(self, taxids):
        """Build a hierarchical tree structure from taxonomy data"""
        with PerformanceMonitor(self.logger):
            tree = defaultdict(lambda: defaultdict(int))
            
            for taxid in tqdm(taxids, desc="Building lineage tree"):
                if taxid in self.taxonomy_dict:
                    info = self.taxonomy_dict[taxid]
                    lineage = info.get('lineage', {})
                    
                    current_path = []
                    for rank in self.rank_order:
                        if rank in lineage:
                            current_path.append(lineage[rank])
                            tree[rank][tuple(current_path)] += 1
            
            return tree
    
    def calculate_diversity_metrics(self, taxids):
        """Calculate various diversity metrics"""
        with PerformanceMonitor(self.logger):
            species_counts = defaultdict(int)
            genus_counts = defaultdict(int)
            
            for taxid in tqdm(taxids, desc="Calculating diversity metrics"):
                if taxid in self.taxonomy_dict:
                    info = self.taxonomy_dict[taxid]
                    lineage = info.get('lineage', {})
                    
                    if 'species' in lineage:
                        species_counts[lineage['species']] += 1
                    if 'genus' in lineage:
                        genus_counts[lineage['genus']] += 1
            
            def shannon_index(counts):
                total = sum(counts.values())
                proportions = [count/total for count in counts.values()]
                return -sum(p * np.log(p) for p in proportions if p > 0)
            
            def simpson_index(counts):
                total = sum(counts.values())
                proportions = [count/total for count in counts.values()]
                return 1 - sum(p*p for p in proportions)
            
            metrics = {
                'species_richness': len(species_counts),
                'genus_richness': len(genus_counts),
                'shannon_diversity_species': shannon_index(species_counts),
                'shannon_diversity_genus': shannon_index(genus_counts),
                'simpson_diversity_species': simpson_index(species_counts),
                'simpson_diversity_genus': simpson_index(genus_counts),
                'total_species': sum(species_counts.values()),
                'total_genera': sum(genus_counts.values())
            }
            
            return metrics

def parse_blast_results(blast_output, qcov_threshold, logger):
    """Parse BLAST results with improved error handling and monitoring"""
    logger.info(f"Parsing BLAST results from '{blast_output}'")
    matched_sequences = defaultdict(list)
    
    try:
        with open(blast_output, 'r') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            for row_num, row in enumerate(tsv_reader, 1):
                if len(row) >= 15:
                    query_id = row[0]
                    subject_id = row[1]
                    percent_identity = float(row[2])
                    alignment_length = int(row[3])
                    query_length = int(row[14])
                    query_coverage = float(row[13])  # Using qcovs from BLAST output
                    
                    if query_coverage >= qcov_threshold:
                        matched_sequences[query_id].append((
                            subject_id, percent_identity, row[12],
                            query_coverage, alignment_length, query_length
                        ))
                else:
                    logger.warning(f"Row {row_num} has insufficient fields: {len(row)}")
        
        logger.info(f"Found {len(matched_sequences)} sequences with matches above thresholds")
        return matched_sequences
        
    except Exception as e:
        logger.error(f"Error parsing BLAST results: {str(e)}")
        raise

def chunk_file(file_path, num_chunks, logger):
    """Split input file into chunks with improved monitoring"""
    logger.info(f"Splitting input file '{file_path}' into {num_chunks} chunks")
    
    try:
        records = list(SeqIO.parse(file_path, "fasta"))
        total_records = len(records)
        chunk_size = total_records // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != num_chunks - 1 else total_records
            
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as chunk_file:
                SeqIO.write(records[start:end], chunk_file.name, "fasta")
                chunks.append(chunk_file.name)
                logger.debug(f"Created chunk {i+1}/{num_chunks}: {chunk_file.name}")
        
        logger.info(f"Successfully split file into {num_chunks} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting file: {str(e)}")
        raise

class TaxonomyVisualizer:
    def __init__(self, figsize=(12, 8), logger=None):
        self.figsize = figsize
        self.logger = logger
        import seaborn as sns  # Ensure seaborn is imported
        sns.set_style('darkgrid')  # Set seaborn style directly

    def plot_all_distributions(self, parsed_data, output_dir):
        """Generate all taxonomy distribution plots"""
        with PerformanceMonitor(self.logger):
            self.plot_rank_distributions(parsed_data, output_dir)
            self.plot_stacked_distribution(parsed_data, output_dir)
            self.plot_species_comparison(parsed_data, output_dir)
            self.plot_network_diagram(parsed_data, output_dir)
            self.plot_diversity_metrics(parsed_data, output_dir)

    def plot_hierarchical_sunburst(self, tree_data):
        """Create a hierarchical sunburst plot with improved design"""
        plt.figure(figsize=(12, 12))
        
        def plot_level(data, start_angle=0, radius=1, level=0):
            total = sum(data.values())
            current_angle = start_angle
            
            for key, value in data.items():
                angle = 2 * np.pi * value / total
                
                # Plot sector
                angles = np.linspace(current_angle, current_angle + angle, 50)
                radii = np.linspace(radius, radius + 0.8, 50)
                r, theta = np.meshgrid(radii, angles)
                
                color = plt.cm.viridis(level / 5)  # Color based on level
                plt.pcolormesh(theta, r, np.ones_like(r), color=color, alpha=0.7)
                
                # Add labels for significant segments
                if value / total > 0.05:  # Label only segments > 5%
                    mid_angle = current_angle + angle/2
                    mid_radius = radius + 0.4
                    x = mid_radius * np.cos(mid_angle)
                    y = mid_radius * np.sin(mid_angle)
                    
                    label = str(key)
                    if len(label) > 20:
                        label = label[:17] + "..."
                    
                    rotation = np.degrees(mid_angle)
                    if rotation > 90 and rotation < 270:
                        rotation += 180
                    
                    plt.text(x, y, label, ha='center', va='center', rotation=rotation)
                
                if isinstance(value, dict):
                    plot_level(value, current_angle, radius + 1, level + 1)
                
                current_angle += angle
        
        # Call the plotting function
        plot_level(tree_data)
        plt.title("Taxonomic Distribution (Sunburst)")
        plt.axis('equal')
        return plt.gcf()

    def plot_diversity_heatmap(self, metrics_by_sample):
        """Create an enhanced heatmap of diversity metrics"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        df = pd.DataFrame(metrics_by_sample).T
        
        # Normalize values for better visualization
        df_norm = (df - df.min()) / (df.max() - df.min())
        
        # Create heatmap
        sns.heatmap(df_norm, annot=df.round(2), cmap='viridis', 
                    ax=ax, fmt='.2f', cbar_kws={'label': 'Normalized Value'})
        
        plt.title("Diversity Metrics Across Samples")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

    def plot_stacked_distribution(self, parsed_data, output_dir):
        """Create stacked bar plot of taxonomic distribution"""
        ranks = ['species', 'genus', 'family']
        all_taxa = set()
        
        for rank in ranks:
            if rank in parsed_data:
                all_taxa.update(parsed_data[rank].keys())
        
        if all_taxa:
            plt.figure(figsize=(15, 8))
            all_taxa = list(all_taxa)
            bottom = np.zeros(len(all_taxa))
            
            for rank in ranks:
                values = [parsed_data[rank].get(taxon, 0) for taxon in all_taxa]
                plt.bar(all_taxa, values, bottom=bottom, label=rank)
                bottom += values
            
            plt.title('Taxonomic Distribution Across Ranks')
            plt.xlabel('Taxa')
            plt.ylabel('Count')
            plt.legend()
            plt.xticks(rotation=90, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'taxonomic_distribution_stacked.png'))
            plt.close()

    def plot_species_comparison(self, parsed_data, output_dir):
        """Create comparative bar plot of species and their higher ranks"""
        if 'species' in parsed_data and parsed_data['species']:
            top_species = dict(sorted(parsed_data['species'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10])
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(top_species))
            width = 0.25
            
            plt.bar(x - width, [parsed_data['species'][sp] for sp in top_species], 
                width, label='Species')
            plt.bar(x, [parsed_data['genus'].get(sp.split()[0], 0) 
                    for sp in top_species], width, label='Genus')
            plt.bar(x + width, [parsed_data['family'].get(sp.split()[0], 0) 
                            for sp in top_species], width, label='Family')
            
            plt.xlabel('Species')
            plt.ylabel('Count')
            plt.title('Top 10 Species Comparison')
            plt.xticks(x, top_species.keys(), rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_10_species_comparison.png'))
            plt.close()

    def plot_diversity_metrics(self, parsed_data, output_dir):
        """Plot diversity metrics visualization"""
        metrics = {}
        ranks = ['species', 'genus', 'family']
        
        for rank in ranks:
            if rank in parsed_data and parsed_data[rank]:
                counts = parsed_data[rank]
                total = sum(counts.values())
                proportions = [count/total for count in counts.values()]
                
                # Calculate diversity metrics
                metrics[f'{rank}_richness'] = len(counts)
                metrics[f'{rank}_shannon'] = -sum(p * np.log(p) for p in proportions if p > 0)
                metrics[f'{rank}_simpson'] = 1 - sum(p*p for p in proportions)
        
        # Create metrics visualization
        plt.figure(figsize=(10, 6))
        x = np.arange(len(ranks))
        width = 0.25
        
        # Plot different metrics
        plt.bar(x - width, [metrics.get(f'{r}_richness', 0) for r in ranks], 
            width, label='Richness')
        plt.bar(x, [metrics.get(f'{r}_shannon', 0) for r in ranks], 
            width, label='Shannon Diversity')
        plt.bar(x + width, [metrics.get(f'{r}_simpson', 0) for r in ranks], 
            width, label='Simpson Diversity')
        
        plt.xlabel('Taxonomic Rank')
        plt.ylabel('Metric Value')
        plt.title('Diversity Metrics by Taxonomic Rank')
        plt.xticks(x, ranks)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diversity_metrics.png'))
        plt.close()
    
    def plot_rank_distributions(self, parsed_data, output_dir):
        """Plot distribution for each taxonomic rank"""
        ranks = ['species', 'genus', 'family']
        
        for rank in ranks:
            if rank not in parsed_data or not parsed_data[rank]:
                self.logger.warning(f"No data for {rank}. Skipping plot.")
                continue
            
            data = Counter(parsed_data[rank])
            top_10 = dict(sorted(data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Bar plot
            plt.figure(figsize=self.figsize)
            bars = plt.bar(top_10.keys(), top_10.values())
            plt.title(f'Top 10 {rank.capitalize()} Distribution')
            plt.xlabel(rank.capitalize())
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height}',
                         ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{rank}_distribution_bar.png'))
            plt.close()

            # Pie chart
            plt.figure(figsize=(10, 10))
            plt.pie(top_10.values(), labels=top_10.keys(), autopct='%1.1f%%',
                   startangle=90)
            plt.axis('equal')
            plt.title(f'Top 10 {rank.capitalize()} Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{rank}_distribution_pie.png'))
            plt.close()
    
    def plot_network_diagram(self, parsed_data, output_dir):
        """Create a network diagram of taxonomic relationships"""
        G = nx.Graph()
        
        # Add nodes and edges
        for rank in ['species', 'genus', 'family']:
            if rank in parsed_data:
                for taxon, count in parsed_data[rank].items():
                    G.add_node(taxon, rank=rank, count=count)
                    
                    if rank == 'species':
                        genus = taxon.split()[0]
                        if genus in parsed_data['genus']:
                            G.add_edge(taxon, genus)
                    elif rank == 'genus':
                        for species in parsed_data['species']:
                            if species.startswith(taxon):
                                G.add_edge(species, taxon)
        
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1000, font_size=8)
        plt.title("Taxonomic Relationships Network")
        plt.savefig(os.path.join(output_dir, 'taxonomy_network.png'))
        plt.close()

def plot_taxonomy(parsed_data, output_dir):
    """Legacy function for backwards compatibility"""
    visualizer = TaxonomyVisualizer()
    logger = logging.getLogger(__name__)
    visualizer.logger = logger
    visualizer.plot_all_distributions(parsed_data, output_dir)

def save_all_visualizations(analyzer, visualizer, taxids, output_dir, logger):
    """Generate and save all visualizations"""
    with PerformanceMonitor(logger):
        logger.info("Generating visualizations")
        
        # Build tree and calculate metrics
        tree = analyzer.build_lineage_tree(taxids)
        metrics = analyzer.calculate_diversity_metrics(taxids)
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate and save plots
        visualizer.plot_hierarchical_sunburst(tree).savefig(
            os.path.join(vis_dir, 'sunburst_plot.png'))
        
        visualizer.plot_diversity_heatmap({'Sample': metrics}).savefig(
            os.path.join(vis_dir, 'diversity_heatmap.png'))
        
        # For plot_network_diagram, we need to prepare parsed_data
        parsed_data = {'tree': tree}
        visualizer.plot_network_diagram(parsed_data, vis_dir)
        
        # Save metrics as JSON
        metrics_file = os.path.join(output_dir, 'diversity_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Visualization generation complete")
        return metrics

def generate_html_report(input_files, metrics, output_dir, report_path, logger):
    """Generate a comprehensive HTML report of the analysis"""
    with PerformanceMonitor(logger):
        logger.info("Generating HTML report")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BLAST_Ripper Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .metric-value {{ font-weight: bold; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BLAST_Ripper Analysis Report</h1>
                <div class="section">
                    <h2>Analysis Summary</h2>
                    <p>Analysis completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Number of input files processed: {len(input_files)}</p>
                </div>
                
                <div class="section">
                    <h2>Diversity Metrics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        for metric, value in metrics.items():
            html_content += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td class="metric-value">{value:.4f if isinstance(value, float) else value}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="visualization">
                        <h3>Taxonomic Distribution (Sunburst)</h3>
                        <img src="visualizations/sunburst_plot.png" alt="Sunburst Plot" style="max-width: 100%;">
                    </div>
                    
                    <div class="visualization">
                        <h3>Diversity Metrics Heatmap</h3>
                        <img src="visualizations/diversity_heatmap.png" alt="Diversity Heatmap" style="max-width: 100%;">
                    </div>
                    
                    <div class="visualization">
                        <h3>Taxonomic Network</h3>
                        <img src="visualizations/taxonomy_network.png" alt="Taxonomic Network" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Files Processed</h2>
                    <table>
                        <tr>
                            <th>File Name</th>
                            <th>Path</th>
                        </tr>
        """
        
        for file_path in input_files:
            file_name = os.path.basename(file_path)
            html_content += f"""
                        <tr>
                            <td>{file_name}</td>
                            <td>{file_path}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated at: {report_path}")

def generate_summary_report(input_files, matched_sequences, taxonomy_dict, output_dir, logger):
    """Generate comprehensive summary report"""
    report_path = os.path.join(output_dir, "summary_report.txt")
    logger.info(f"Generating summary report at {report_path}")
    
    try:
        with open(report_path, 'w') as f:
            # Header
            f.write("BLAST_Ripper Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Input Statistics
            f.write("1. Input Statistics\n")
            f.write("-" * 20 + "\n")
            total_sequences = sum(len(list(SeqIO.parse(file, "fasta"))) for file in input_files)
            f.write(f"Total input files: {len(input_files)}\n")
            f.write(f"Total sequences analyzed: {total_sequences:,d}\n")
            
            # BLAST Results
            f.write("\n2. BLAST Results\n")
            f.write("-" * 20 + "\n")
            matched_count = len(matched_sequences)
            match_rate = (matched_count/total_sequences)*100
            f.write(f"Sequences with matches: {matched_count:,d}\n")
            f.write(f"Match rate: {match_rate:.2f}%\n")
            f.write(f"Sequences without matches: {total_sequences - matched_count:,d}\n")
            
            # Quality Statistics
            f.write("\n3. Quality Statistics\n")
            f.write("-" * 20 + "\n")
            identities = [match[1] for matches in matched_sequences.values() for match in matches]
            coverages = [match[3] for matches in matched_sequences.values() for match in matches]
            
            f.write(f"Average identity: {np.mean(identities):.2f}%\n")
            f.write(f"Average coverage: {np.mean(coverages):.2f}%\n")
            f.write(f"Identity range: {min(identities):.2f}% - {max(identities):.2f}%\n")
            f.write(f"Coverage range: {min(coverages):.2f}% - {max(coverages):.2f}%\n")
            
            # Taxonomic Summary
            f.write("\n4. Taxonomic Summary\n")
            f.write("-" * 20 + "\n")
            for rank in ['species', 'genus', 'family']:
                f.write(f"\nTop 10 {rank.capitalize()}:\n")
                rank_counts = Counter()
                for matches in matched_sequences.values():
                    for match in matches:
                        taxid = match[2]  # taxid from match tuple
                        if taxid in taxonomy_dict:
                            if rank == 'species':
                                name = taxonomy_dict[taxid]['name']
                            else:
                                name = taxonomy_dict[taxid]['lineage'].get(rank, 'Unknown')
                            rank_counts[name] += 1
                
                for name, count in rank_counts.most_common(10):
                    percentage = (count/matched_count)*100
                    f.write(f"{name}: {count:,d} ({percentage:.2f}%)\n")
            
            f.write("\n5. Analysis Parameters\n")
            f.write("-" * 20 + "\n")
            f.write("BLAST database: nt\n")
            f.write(f"Query coverage threshold: {args.qcov_threshold}%\n")
            f.write(f"Identity threshold: {args.identity_threshold}%\n")
            
        logger.info("Summary report generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        return False

def generate_detailed_report(matched_sequences, taxonomy_dict, output_dir, logger):
    """Generate detailed TSV report with extended statistics"""
    report_path = os.path.join(output_dir, "detailed_report.tsv")
    logger.info(f"Generating detailed report at {report_path}")
    
    try:
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'Sequence_ID',
                'Match_Count',
                'Best_Identity',
                'Best_Coverage',
                'Species',
                'Genus',
                'Family',
                'Alignment_Length',
                'Query_Length'
            ])
            
            for seq_id, matches in matched_sequences.items():
                best_match = max(matches, key=lambda x: (x[1], x[3]))  # Sort by identity then coverage
                taxid = best_match[2]
                taxonomy = get_taxonomy_info(taxid, taxonomy_dict)
                
                writer.writerow([
                    seq_id,
                    len(matches),
                    f"{best_match[1]:.2f}",
                    f"{best_match[3]:.2f}",
                    taxonomy['species'],
                    taxonomy['genus'],
                    taxonomy['family'],
                    best_match[4],  # Alignment length
                    best_match[5]   # Query length
                ])
        
        logger.info("Detailed report generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating detailed report: {str(e)}")
        return False

def generate_quality_plots(matched_sequences, output_dir, logger):
    """Generate enhanced quality control plots"""
    try:
        # Identity distribution
        plt.figure(figsize=(12, 6))
        identities = [match[1] for matches in matched_sequences.values() for match in matches]
        plt.hist(identities, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sequence Identities', fontsize=12, pad=20)
        plt.xlabel('Identity (%)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'identity_distribution.png'))
        plt.close()
        
        # Coverage distribution
        plt.figure(figsize=(12, 6))
        coverages = [match[3] for matches in matched_sequences.values() for match in matches]
        plt.hist(coverages, bins=50, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Query Coverage', fontsize=12, pad=20)
        plt.xlabel('Coverage (%)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'coverage_distribution.png'))
        plt.close()
        
        # Identity vs Coverage scatter plot
        plt.figure(figsize=(12, 6))
        plt.scatter(identities, coverages, alpha=0.5, c='blue', s=20)
        plt.title('Identity vs Coverage', fontsize=12, pad=20)
        plt.xlabel('Identity (%)')
        plt.ylabel('Coverage (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'identity_vs_coverage.png'))
        plt.close()
        
        logger.info("Quality control plots generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating quality plots: {str(e)}")
        return False

def generate_report_index(output_dir, logger):
    """Generate HTML index page for all reports"""
    index_path = os.path.join(output_dir, "analysis_report.html")
    logger.info(f"Generating report index at {index_path}")
    
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BLAST_Ripper Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .report-link {{ color: #2c3e50; text-decoration: none; }}
                .report-link:hover {{ color: #3498db; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BLAST_Ripper Analysis Report</h1>
                <div class="section">
                    <h2>Analysis Summary</h2>
                    <p>Analysis completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h3>Available Reports</h3>
                    <ul>
                        <li><a href="summary_report.txt" class="report-link">Summary Report</a></li>
                        <li><a href="detailed_report.tsv" class="report-link">Detailed Analysis Report</a></li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Quality Control Visualizations</h2>
                    <div class="visualization">
                        <h3>Identity Distribution</h3>
                        <img src="identity_distribution.png" alt="Identity Distribution" style="max-width: 100%;">
                    </div>
                    
                    <div class="visualization">
                        <h3>Coverage Distribution</h3>
                        <img src="coverage_distribution.png" alt="Coverage Distribution" style="max-width: 100%;">
                    </div>
                    
                    <div class="visualization">
                        <h3>Identity vs Coverage</h3>
                        <img src="identity_vs_coverage.png" alt="Identity vs Coverage" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Taxonomic Analysis</h2>
                    <div class="visualization">
                        <h3>Species Distribution</h3>
                        <img src="species_distribution_pie.png" alt="Species Distribution" style="max-width: 100%;">
                    </div>
                    
                    <div class="visualization">
                        <h3>Taxonomic Distribution</h3>
                        <img src="taxonomic_distribution_stacked.png" alt="Taxonomic Distribution" style="max-width: 100%;">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        logger.info("Report index generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating report index: {str(e)}")
        return False

def cleanup_temp_files(output_dir, logger):
    """Clean up temporary files after processing"""
    with PerformanceMonitor(logger):
        try:
            # Clean up temporary chunks
            temp_files = [f for f in os.listdir(output_dir) if f.endswith('.temp')]
            for temp_file in temp_files:
                os.remove(os.path.join(output_dir, temp_file))
            
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLAST_Ripper: Parallel BLAST processing with taxonomy analysis",
        add_help=False
    )
    
    # Original arguments
    parser.add_argument('-in', dest='input_folder', required=True,
                      help="Input folder containing FASTA files")
    parser.add_argument('-out', dest='output_folder', required=True,
                      help="Output folder to store results")
    parser.add_argument('-db', dest='db', required=True,
                      help="Path to the BLAST database")
    parser.add_argument('-taxdb', dest='taxdb_dir', required=True,
                      help="Path to taxonomy database files")
    parser.add_argument('-t', '--threads', type=int, default=4,
                      help="Number of threads per BLAST process")
    parser.add_argument('-n', '--num_chunks', type=int, default=8,
                      help="Number of chunks to split each input file into")
    parser.add_argument('-shd', '--use-ramdisk', action='store_true',
                      help="Use /dev/shm for temporary files")
    parser.add_argument('-taxids', nargs='+',
                      help="List of taxids to filter results (optional)")
    parser.add_argument('-qcov', '--qcov_threshold', type=float, default=80,
                      help="Query coverage threshold for considering a match (default: 80)")
    parser.add_argument('-identity', '--identity_threshold', type=float, default=90,
                      help="Identity threshold (default: 90)")
    parser.add_argument('-h', '--help', action=HelpAction, nargs=0,
                      help="Show this help message and exit")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_folder, exist_ok=True)
        
        # Setup logging
        logger = setup_logging(args.output_folder)
        logger.info("Starting BLAST_Ripper pipeline")
        
        # Initialize performance monitoring
        with PerformanceMonitor(logger):
            # Initialize progress bar
            total_steps = 5
            progress = tqdm(total=total_steps, desc="Overall Progress")
            
            # Parse taxonomy database
            logger.info("Parsing taxonomy database")
            taxonomy_dict = parse_local_taxonomy(args.taxdb_dir, logger)
            progress.update(1)
            
            # Initialize analyzers
            analyzer = TaxonomyAnalyzer(taxonomy_dict, logger)
            visualizer = TaxonomyVisualizer(logger=logger)
            
            # Get input files
            input_files = [
                os.path.join(args.input_folder, f) 
                for f in os.listdir(args.input_folder) 
                if os.path.isfile(os.path.join(args.input_folder, f)) 
                and f.endswith(('.fasta', '.fa'))
            ]
            
            logger.info(f"Found {len(input_files)} input files")
            
            # Process each file
            all_taxids = set()
            matched_sequences_all = defaultdict(list)
            for file_path in input_files:
                logger.info(f"Processing file: {file_path}")
                file_taxids, matched_sequences = process_file(
                    file_path=file_path,
                    output_dir=args.output_folder,
                    db=args.db,
                    num_chunks=args.num_chunks,
                    threads=args.threads,
                    use_ramdisk=args.use_ramdisk,
                    taxids=args.taxids,
                    taxonomy_dict=taxonomy_dict,
                    qcov_threshold=args.qcov_threshold,
                    logger=logger
                )
                all_taxids.update(file_taxids)
                for key, value in matched_sequences.items():
                    matched_sequences_all[key].extend(value)
                        
            progress.update(1)
            
            # Perform global analysis
            logger.info("Performing global taxonomy analysis")
            metrics = save_all_visualizations(
                analyzer=analyzer,
                visualizer=visualizer,
                taxids=all_taxids,
                output_dir=args.output_folder,
                logger=logger
            )
            progress.update(1)
            
            # Generate reports
            logger.info("Generating analysis reports")
            generate_summary_report(
                input_files=input_files,
                matched_sequences=matched_sequences_all,
                taxonomy_dict=taxonomy_dict,
                output_dir=args.output_folder,
                logger=logger
            )
            generate_detailed_report(
                matched_sequences=matched_sequences_all,
                taxonomy_dict=taxonomy_dict,
                output_dir=args.output_folder,
                logger=logger
            )
            generate_quality_plots(
                matched_sequences=matched_sequences_all,
                output_dir=args.output_folder,
                logger=logger
            )
            generate_report_index(
                output_dir=args.output_folder,
                logger=logger
            )
            progress.update(1)
            
            # Generate final report
            logger.info("Generating final report")
            report_path = os.path.join(args.output_folder, 'final_report.html')
            generate_html_report(
                input_files=input_files,
                metrics=metrics,
                output_dir=args.output_folder,
                report_path=report_path,
                logger=logger
            )
            progress.update(1)
            
            # Cleanup
            logger.info("Cleaning up temporary files")
            cleanup_temp_files(args.output_folder, logger)
            progress.update(1)
            
            logger.info("Pipeline completed successfully")
            progress.close()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure progress bar is closed
        if 'progress' in locals():
            progress.close()
