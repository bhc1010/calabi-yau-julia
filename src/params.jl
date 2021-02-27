#
#  Parameters for data I/O
TOTAL_DATA = 473_800_775
TOTAL_NUM_DB = 474
DIM = 4
SAMPLE_SIZE = 1_000_000
DB_PATH = "/media/share/Dev/CalabiYau/data/polytopes_db_4d"
PATH = "/home/oppenheimer/Dev/calabiyau/trained"
#
#  Parameters for model building
MAX_SIZE = 34
CLASS_NUM = 961
CLASSES = [x for x in 1:CLASS_NUM]
BATCH_SIZE = 1000
ONTOLOGY = "euler"
DATA_ID = 2

if ONTOLOGY == "h11"
    ontology_index = 2
elseif ONTOLOGY == "h21"
    ontology_index = 3
elseif ONTOLOGY == "euler"
    ontology_index = 4
end

## Add set helper functions

SetDBId(id::Integer) = global DATA_ID = id;