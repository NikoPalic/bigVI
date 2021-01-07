from reuters_parser import parse_reuters

# List of filenames
filenames = ["data/reuters21578/reut2-000.sgm", "data/reuters21578/reut2-001.sgm",
             "data/reuters21578/reut2-002.sgm", "data/reuters21578/reut2-003.sgm",
             "data/reuters21578/reut2-004.sgm", "data/reuters21578/reut2-005.sgm",
             "data/reuters21578/reut2-006.sgm", "data/reuters21578/reut2-007.sgm",
             "data/reuters21578/reut2-008.sgm","data/reuters21578/reut2-009.sgm",
             "data/reuters21578/reut2-010.sgm", "data/reuters21578/reut2-011.sgm",
             "data/reuters21578/reut2-012.sgm", "data/reuters21578/reut2-013.sgm",
             "data/reuters21578/reut2-014.sgm","data/reuters21578/reut2-015.sgm",
             "data/reuters21578/reut2-016.sgm", "data/reuters21578/reut2-017.sgm",
             "data/reuters21578/reut2-018.sgm","data/reuters21578/reut2-019.sgm",
             "data/reuters21578/reut2-020.sgm", "data/reuters21578/reut2-021.sgm"
             ]
full_classes = []
full_corpus = []

for file in filenames:
    classes, corpus = parse_reuters(file,"grain")
    full_classes.append(classes)
    full_corpus.append(corpus)
    
