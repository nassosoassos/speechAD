###########################################################################
#
# File:         parseLDCcatalog.py
# Date:         Fri 20 Mar 2009 12:38
# Author:       Ken Basye
# Description:  Tools for parsing the LCD catalog in XML format.
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
###########################################################################

"""
    >>> module_dir, module_name = os.path.split(__file__)

    A version of the LDC XML file:
    >>> test_file = "ldc_catalog_2009_3_30.xml"
    >>> test_file = os.path.join(module_dir, test_file)

    >>> results = parse_catalog(test_file)
    >>> len(results)
    433

    # For some reason, LDC has removed quite a few language tags lately.
    >>> no_lang = tuple([item for item in results if item[2] is None])
    >>> len(no_lang)
    27
    
    >>> for item in no_lang[0:4]: print item[0:2]
    ('LDC2000T53', 'Voice of America (VOA) Broadcast News Czech Transcript  Corpus')
    ('LDC2001S16', 'Grassfields Bantu Fieldwork: Ngomba Tone Paradigms')
    ('LDC2001T60', 'Syllable-Final /s/ Lenition')
    ('LDC2001T61', 'CALLHOME Spanish Dialogue Act Annotation')

    >>> languages = set((item[2] for item in results))
    >>> len(languages)
    30
    
    >>> test_dir_name = "test_dir"
    >>> test_dir_name = os.path.join(module_dir, test_dir_name)
    >>> results, unlisted = _keep_only_available_datasets(results, test_dir_name)
    >>> len(results)
    2
    >>> results[0][0:2]
    ('LDC2004S13', 'Fisher English Training Speech Part 1 Speech')

    >>> table_str = generate_twiki_table_for_ldc_data(results)
    >>> print table_str
    | *Year* | *Catalog ID* | *Type* | *Language* | *Title* | 
    | 2004 | [[http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2004S13][LDC2004S13]] | Speech | English | Fisher English Training Speech Part 1 Speech | 
    | 2009 | [[http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2009T01][LDC2009T01]] | Text | English | English CTS Treebank with Structural Metadata | 
    <BLANKLINE>

##     Make sure we can get the catalog directly from the LDC web server.  The length of the results
##     will be a slow-moving target
##     >>> results = parse_catalog(STD_URL)
##     >>> len(results)
##     435

##     Check on current state of language tags - probably another slow-moving target
##     >>> no_lang = tuple([item for item in results if item[2] is None])
##     >>> len(no_lang)
##     35
"""

from __future__ import with_statement
import xml.etree.ElementTree
import os
import os.path
import sys
import re
import urllib
import urlparse
import operator
from unlistedLDCinfo import _UNLISTED_INFO_DICT

STD_URL = "http://projects.ldc.upenn.edu/OLAC/ldc_catalog.xml"

# The usual pattern is 'LDC'<YEAR><TYPE><NUMBER> but there are two styles for year (two-digit until
# 99, then four-digit from 2000 on.  NUMBER is usually just digits, but there are some quirky
# exceptions :-<.
_LDC_CAT_ID_RE = re.compile("^LDC(9[3456789]|20\d\d)[STLRVE]\d+((U01)|([ABCD]?)|(B?-[12345]))$")
def _valid_ldc_catalog_id(cat_id):
    return _LDC_CAT_ID_RE.match(cat_id) is not None


def _year_and_type_from_catalog_id(cat_id):
    _TYPEMAP = {'S': 'Speech',
                'T': 'Text',
                'L': 'Linguistic Tool',
                'R': 'Restricted',
                'V': 'Video',
                'E': 'Evaluation',
                }
    
    if not _valid_ldc_catalog_id(cat_id):
        raise ValueError('Bad LDC catalog ID %s' % (cat_id,))
    if cat_id[3] == '9':
        year = '19' + cat_id[3:5]
        type_char = cat_id[5]
    elif cat_id[3:5] == '20':
        year = cat_id[3:7]
        type_char = cat_id[7]
    else:
        assert False
    assert _TYPEMAP.has_key(type_char)
    return year, _TYPEMAP[type_char]

    
def parse_catalog(source):
    """
    Parse the XML file provided by LDC which gives lots of information about
    each of their publications.  Note, though, that there are some interesting
    'holes' in this meta-data.

    *source* should be a string, and may be either the name of a file or a URL which should point
    directly to the LDC XML file.  
    Returns a tuple of tuples, where each interior tuple starts with
    the LDC catalog number as a string.
    """

    url_tuple = urlparse.urlparse(source)
    # See if the argument is a real URL or just a filename.  The urlparse function will create a
    # tuple; the second element of that tuple is the network location, so if we have one of those,
    # we'll treat this as a real URL.
    if url_tuple[1] == '':
        source_file = open(source)
    else:
        source_file = urllib.urlopen(source)

    tree = xml.etree.ElementTree.parse(source_file)
    source_file.close()

    # Set up some useful constants for the parse

    _RECORD_TAG = "{http://www.openarchives.org/OAI/2.0/}record"
    _METADATA_TAG = "{http://www.openarchives.org/OAI/2.0/}metadata"
    _OLAC_TAG = "{http://www.language-archives.org/OLAC/1.1/}olac"
    _ID_TAG = "{http://purl.org/dc/elements/1.1/}identifier"
    _TITLE_TAG = "{http://purl.org/dc/elements/1.1/}title"
    _DESCRIPTION_TAG = "{http://purl.org/dc/elements/1.1/}description"
    _LANGUAGE_TAG = "{http://purl.org/dc/elements/1.1/}language"

    # pairs of prefix, optional for parsing description fields
    _DESCRIPTION_FIELDS = (("Data source: ", True),)

    all_records = tuple(tree.getiterator(_RECORD_TAG))

    results = list()
    for record in all_records:
        # We have to descend through two levels to get to the real record.  I
        # may be doing this more carefully than is really necessary.
        sub_record = record.find(_METADATA_TAG)
        if sub_record is None:
            raise ValueError("Failed to find metadata subelement in record")
        sub_record = sub_record.find(_OLAC_TAG)
        if sub_record is None:
            raise ValueError("Failed to find an olac subelement in metadata")
        record = sub_record
        
        # There are two _ID_TAGs in each record; the first one is the one we want
        cat_id = record.findtext(_ID_TAG)
        if cat_id is None or not cat_id.startswith("LDC"):
            raise ValueError("Failed to find a valid catalog id")
        year, data_type = _year_and_type_from_catalog_id(cat_id)
        title = record.findtext(_TITLE_TAG)
        if title is None or title == '':
            raise ValueError("Failed to find a valid title")
        # There are several _DESCRIPTION_TAGs in each sub_record, and we only care
        # about some of them
        description_data = list()
        found_fields = set()
        for prefix, optional in _DESCRIPTION_FIELDS:
            for desc in record.getiterator(_DESCRIPTION_TAG):
                if desc.text.startswith(prefix):
                    description_data.append(desc.text[len(prefix):])
                    found_fields.add(prefix)
            if not optional and prefix not in found_fields:
                raise ValueError("Failed to find a description with prefix: %s" % prefix)
        
        language = record.findtext(_LANGUAGE_TAG)
        if language == '':
            raise ValueError("Found empty language tag")
        results.append((cat_id, title, language, year, data_type) + tuple(description_data))
    return tuple(results)


def _keep_only_available_datasets(dataset_info, corpora_location=None):
    """
    Filter information about datasets (assumed to be in the form generated by
    parse_catalog(), see above) and keep only information about those datasets
    we actually have.  This is done by comparing directory names with catalog
    IDs, which works for us because we use catalog IDs as directory names.
    """
    _STD_LDC_CORPORA_LOCATION = "/export/common/data/corpora/LDC"
    if corpora_location is None:
        corpora_location = _STD_LDC_CORPORA_LOCATION
    dirset = set(os.listdir(corpora_location))
    kept = tuple((info for info in dataset_info if info[0] in dirset))
    listed_ids = set((info[0] for info in dataset_info))
    missing = tuple((dir for dir in (dirset - listed_ids) if _valid_ldc_catalog_id(dir)))
    return kept, missing
    

def _generate_info_for_missing_items(missing, warn_if_not_found=True):
    """
    Generate tuples in the style of parseLDCcatalog() for items given only their catalog numbers.
    This means the title, language, and source fields will be 'unknown'.

    >>> missing = ('LDC2004E24', 'LDC97T1')
    >>> for info in _generate_info_for_missing_items(missing, warn_if_not_found=False):
    ...   print info
    ('LDC2004E24', 'RT-04F MDE Consistency Study', 'English', '2004', 'Evaluation', 'Unknown')
    ('LDC97T1', 'Unknown', 'Unknown', '1997', 'Text', 'Unknown')
    """
    results = list()
    for cat_id in missing:
        year, data_type = _year_and_type_from_catalog_id(cat_id)
        if not _UNLISTED_INFO_DICT.has_key(cat_id):
            title, language, description_data = ("Unknown", "Unknown", ("Unknown",))
            if warn_if_not_found:
                print >> sys.stderr, "Warning - no information found for unlisted item %s" % (cat_id,)
        else:
            title, language, description_data = _UNLISTED_INFO_DICT[cat_id]
        results.append((cat_id, title, language, year, data_type) + tuple(description_data))
    return tuple(results)
        
def generate_twiki_table_for_ldc_data(dataset_info, missing=None, format_spec=None):
    _SEP = ' | ' 
    # Keep this tuple in the order you want the fields of the table to appear
    # in.  The first of each pair is the title for the column, the second is the
    # index in the information tuple where the datum is to be found (see
    # parse_catalog).
    _STD_FORMAT_SPEC= (('Year', 3), ('Catalog ID', 0), ('Type', 4), ('Language', 2), ('Title', 1))

    if format_spec is None:
        format_spec = _STD_FORMAT_SPEC

    # permute data into order given by format spec and sort in this order
    getter = operator.itemgetter(*[field[1] for field in format_spec])
    sorted = [getter(info) for info in dataset_info]
    sorted.sort()

    # Generate header line in Twiki format
    ret = _SEP.lstrip()
    for field_name, not_used in format_spec:
        assert _SEP.strip() not in field_name
        ret += ('*%s*' % field_name) + _SEP
    ret += '\n'

    wikiword_re = re.compile(r"\(?\b([A-Z]+[a-z]+[A-Z]+\w)\)?")
    def mark_wiki_word(word):
        return '!' + word if word[0] != '(' else '(!' + word[1:]
    def wiki_word(word):
        return wikiword_re.match(word) is not None
            
    def format_one_line(info, catalog_id_idx):
        line = _SEP.lstrip()
        for idx, field in enumerate(info):
            assert field is None or _SEP.strip() not in field
            if field is not None and idx != catalog_id_idx:
                field = " ".join([mark_wiki_word(word) if wiki_word(word) else word for word in field.split()])
            if idx != catalog_id_idx:
                line += str(field) + _SEP
            else:
                line += '[[http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=%s][%s]]' % (field, field) + _SEP
        line += '\n'
        return line
        
    # Generate body lines for table
    for info in sorted:
        ret += format_one_line(info, 1)

    return ret

def main(outfilename):
    results = parse_catalog(STD_URL)
    listed_holdings_info, unlisted_holdings = _keep_only_available_datasets(results)
    unlisted_holdings_info = _generate_info_for_missing_items(unlisted_holdings)
    holdings_info = listed_holdings_info + unlisted_holdings_info
    table_str = generate_twiki_table_for_ldc_data(holdings_info)
    with open(outfilename, "w") as outfile:
        outfile.write(table_str)
        
def usage(prog_name):
    print("Usage: %s [-o outfile]" % (prog_name,))
    print("With no arguments, run doctests; with outfile, generate a new twiki table in outfile")
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        from onyx import onyx_mainstartup
        onyx_mainstartup()
    elif len(sys.argv) == 3 and sys.argv[1] == '-o':
        main(sys.argv[2])
    else:
        usage(sys.argv[0])
    



