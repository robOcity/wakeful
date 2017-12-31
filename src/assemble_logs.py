import gzip
import glob
import os.path
import sys
import click


@click.command()
@click.option('--src_dir', help='Directory to recursively search for log files')
@click.option('--dest_dir', help='Path to output directory that may be relative or absolute')
@click.option('--log_type', default='dns', help='Type of Bro log file to process')
def assemble_logs(src_dir, dest_dir, log_type):
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
    with open(dest_file, 'ab') as outfile:
        for src_name in glob.iglob(path, recursive=True):
            with gzip.open(src_name, 'rb') as infile:
                for line in infile:
                    outfile.write(line)
                    num_lines += 1
            f_state = f'outfile: {str({outfile})}'
            num_files += 1
    # click side-effect prevents this running in main, or printing the returned values too. WEIRD!!
    click.echo(f'{num_files} files with {num_lines} lines processed')
    return num_files, num_lines


def remove_existing_file(filepath):
    """Removes the file if it exists"""
    if os.path.exists(filepath):
        os.remove(filepath)


if __name__ == '__main__':
    num_files, num_lines = assemble_logs()
