# Data preparation

The main source of our data is the Xeno-Canto library. [Xeno-Canto](https://www.xeno-canto.org/) is a website dedicated to sharing bird sounds from all over the world. They were so kind to let us have free access to their database.

1. Getting all the species

First we [scraped](/get_taxonomy_xeno-canto.py) the Xeno-Canto website to get the taxonomy of all the species found on the website and saved it in a [json file](/taxonomy.txt).

2. Creating a sqlite database

Secondly we created a lightweight SQLite database with two tables: 

* taxonomy
* recordings

The taxonomy table contains all the bird species (Order, Family, Genus, Species). We inserted the species extracted in the first step in this table. The recordings table will contain the metadata of all the recordings found in the database.

3. Getting the recordings metadata

This [script](https://www.xeno-canto.org/api/2/recordings?query=foo) uses the [Xeno-Canto api](https://www.xeno-canto.org/api/2/recordings?query=foo) to get the metadata of all the recordings in their database and save it to our SQLite database.

At this point we decided to focus on the bird species that are found in Germany.

We [scraped](/german_bird_list.py) the [wikipedia page](https://commons.wikimedia.org/wiki/Liste_der_V%C3%B6gel_Deutschlands) for german species and saved it as a boolean field in the taxonomy database table.

4. Downloading the recordings

This [script](/script_download_files_threaded.py) downloads all mp3 files from german species to the storage location. We use threads to drastically speed up this process.

Once downloaded, we flagged all the downloaded recordings in the database.

5. 