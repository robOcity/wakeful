import gzip
import glob
import os.path
import sys
import click
import pandas as pd
from bat.bro_log_reader import BroLogReader


@click.command()
@click.option('--src_dir', help='Directory to recursively search for log files')
@click.option('--dest_dir', help='Path to output directory that may be relative or absolute')
@click.option('--log_type', default='dns', help='Type of Bro log file to process')
def main(src_dir, dest_dir, log_type):
    """
    Ingests Bro Network Security Monitor logs and concatentates them into a
    a single output file suitible.  Optionally decompresses the files before
    appending them together.
    """

    # recursively get all the file in directoreis under src_dir
    path = os.path.join(src_dir, '*', log_type.lower() + '**.gz')
    num_files, num_lines = 0, 0
    dest_file = os.path.join(dest_dir, log_type.lower() + '_all.log')
    remove_existing_file(dest_file)

    # recursively get all the file in directoreis under src_dir
    path = os.path.join(src_dir, '*', log_type.lower() + '**.gz')

    # create the dataframe
    bro_log_df = pd.DataFrame()

    for gz_filename in glob.iglob(path, recursive=True):
        # unpack the data so bat can process it
        log_filename = decompress(gz_filename)

        # TODO: add capability to process gzipped files to bat or file-handles
        reader = BroLogReader(log_filename)

        # store dictionaries, one for each line in the logs
        log_data = []
        for row in reader.readrows():
            log_data.append(row)
            num_lines += 1
            num_files += 1

        bro_log_df.append(log_data)
        # click side-effect prevents this running in main, or printing the returned values too. WEIRD!!
        click.echo(f'Bro log data frame shape: {bro_log_df.shape}')

    return bro_log_df


def decompress(gz_filename):
    """Decompress the gzipped archive file and write to disk."""
    # remove the ".gz"
    log_filename = os.path.splitext(gz_filename)[0]
    with open(log_filename, 'wb') as outfile:
        with gzip.open(gz_filename, 'rb') as infile:
            for line in infile:
                outfile.write(line)
    return log_filename


def remove_existing_file(filepath):
    """Removes the file if it exists"""
    if os.path.exists(filepath):
        os.remove(filepath)


def save_to_file(df, path):
    """Save the dataframe as a comma separated values file"""
    df.to_csv(path)


if __name__ == '__main__':
    main()
