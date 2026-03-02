"""
Script per scaricare e integrare FastText pre-trainati di Facebook
Questi sono trainati su Common Crawl con 2M+ vocabolario
"""
import os
import requests
from tqdm import tqdm
import gzip
import shutil

BASE_DIR = '/home/luca/PycharmProjects/arianna-assistant'

def download_file(url, dest_path):
    """Download file con progress bar"""
    print(f"📥 Download da: {url}")
    print(f"📁 Destinazione: {dest_path}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    print("✓ Download completato!")

def decompress_gz(gz_path, output_path):
    """Decomprimi file .gz"""
    print(f"📦 Decompressione: {gz_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"✓ Decompresso in: {output_path}")

def main():
    print("="*80)
    print("DOWNLOAD FASTTEXT PRE-TRAINATI (ITALIANO)")
    print("="*80)

    models_dir = os.path.join(BASE_DIR, 'models')

    # URL del modello FastText italiano di Facebook
    # Nota: cc.it.300.bin è ~7GB! Usiamo la versione più piccola
    # cc.it.300.vec.gz è ~4GB (solo vettori, non il modello completo)

    print("\n⚠️  ATTENZIONE: Il file è molto grande (~4-7 GB)")
    print("    Assicurati di avere spazio su disco e connessione stabile")

    choice = input("\nScarica versione:\n  1. Modello completo (.bin, ~7GB)\n  2. Solo vettori (.vec, ~4GB)\n  3. Annulla\n\nScelta [1/2/3]: ")

    if choice == '1':
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz"
        gz_path = os.path.join(models_dir, 'cc.it.300.bin.gz')
        output_path = os.path.join(models_dir, 'fasttext_pretrained.bin')

        download_file(url, gz_path)
        decompress_gz(gz_path, output_path)
        os.remove(gz_path)

        print(f"\n✓ Modello salvato in: {output_path}")
        print(f"\n📝 Prossimo step:")
        print(f"   Modifica intent_classifier.py per usare: {output_path}")

    elif choice == '2':
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz"
        gz_path = os.path.join(models_dir, 'cc.it.300.vec.gz')
        output_path = os.path.join(models_dir, 'fasttext_pretrained.vec')

        download_file(url, gz_path)
        decompress_gz(gz_path, output_path)
        os.remove(gz_path)

        print(f"\n✓ Vettori salvati in: {output_path}")
        print(f"\n📝 Prossimo step:")
        print(f"   Converti .vec in .bin oppure carica manualmente con gensim")

    else:
        print("\n❌ Download annullato")
        return

    print("\n" + "="*80)

if __name__ == "__main__":
    main()

