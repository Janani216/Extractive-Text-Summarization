import glob

root_dir = '/Users/shreyaballijepalli/Downloads/cnn/stories'


def parse_document(lines):
    highlight_index = lines.index('@highlight')
    doc_lines = lines[:highlight_index]
    summary_lines = lines[highlight_index:]
    doc_lines = list(filter(lambda doc_line: len(doc_line) > 0, doc_lines))
    summary_lines = list(filter(lambda sum_line: sum_line != '@highlight' and len(sum_line) > 0, summary_lines))
    return "\n".join(doc_lines), ".\n".join(summary_lines)


def write_to_file(document, summary, doc_count):
    doc_file_name = 'cnn/document_{}.txt'.format(doc_count)
    summary_file_name = 'cnn/summary_{}.txt'.format(doc_count)
    with open(doc_file_name, 'w') as f:
        f.write(document)
    with open(summary_file_name, 'w') as f:
        f.write(summary)


if __name__ == '__main__':
    story_files = glob.glob(root_dir + "/*.story")
    doc_count = 0
    for file in story_files:
        lines = open(file).read().splitlines()
        document, summary = parse_document(lines)
        doc_count += 1
        write_to_file(document, summary, doc_count)
