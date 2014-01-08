###########################################################################
#
# File:         objectset.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Indexed set of runtime objects constructed from text serialization
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 The Johns Hopkins University
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
Indexed set of runtime objects constructed from text serialization

"""


# Yaml version

from onyx.textdata.yamldata import YamldataReader, YamldataWriter

class IndexedObjectSet(object):
    """
    Constructs a sequence of objects based on a text specification.

    Example constructing some signal processing objects.

    >>> doc = '''
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : IndexedObjectSet
    ...   __onyx_yaml__stream_version : '0'
    ...   __onyx_yaml__stream_options : implicit_index=True
    ...   
    ... -
    ...   # format for IndexedObjectSet, where presence of index field depends on value of implicit_index
    ...   # [index] module factory serial_version=N remaining-args-string
    ...   - onyx.signalprocessing.spectrum PreEmphasis serial_version=0
    ...   - onyx.signalprocessing.window Sliding serial_version=0 length=25000*usec  shift=10000*usec
    ...   - onyx.signalprocessing.window Hamming serial_version=0
    ...   - onyx.signalprocessing.window Padding serial_version=0 length=power_of_two
    ...   - onyx.signalprocessing.spectrum Fft serial_version=0
    ...   - onyx.signalprocessing.window Truncate serial_version=0 length=half_plus_one
    ...   - onyx.signalprocessing.scale Abs serial_version=0
    ...   # Bark band edges (in Hz):
    ...   #   20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ...   # Bark band centers (in Hz):
    ...   #   50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500
    ...   # As in Sphinx4, we interleave the edges and centers. See:
    ...   #   http://cmusphinx.sourceforge.net/sphinx4/javadoc/edu/cmu/sphinx/frontend/frequencywarp/MelFrequencyFilterBank.html
    ...   - onyx.signalprocessing.spectrum MelFilter serial_version=0 units=hz 20 50 100 150 200 250 300 350 400 450 510 570 630 700 770 840 920 1000 1080 1170 1270 1370 1480 1600 1720 1850 2000 2150 2320 2500 2700 2900 3150 3400 3700 4000 4400 4800 5300 5800 6400 7000 7700 8500 9500 10500 12000 13500 15500
    ...   
    ...   - onyx.signalprocessing.scale Log serial_version=0 base=10 scale=10
    ...   - onyx.signalprocessing.spectrum Dct serial_version=0  
    ...   - onyx.signalprocessing.window Truncate serial_version=0 length=13
    ...   - onyx.signalprocessing.delta Delta serial_version=0
    ...   - onyx.signalprocessing.delta DeltaDelta serial_version=0
    ...   - onyx.dataflow.graph Join serial_version=0
    ...   '''

    >>> objects = IndexedObjectSet(doc)
    >>> items = tuple(objects)
    >>> len(items)
    14
    >>> s = items[1][1]
    >>> s
    Sliding(['serial_version=0', 'length=25000*usec', 'shift=10000*usec'])

    >>> s = items[8][1]
    >>> s
    Log(['serial_version=0', 'base=10', 'scale=10'])
    >>> s.serialize()
    ('onyx.signalprocessing.scale', 'Log', '0', 'base=10', 'scale=10')

    """

    STREAM_TYPE = 'IndexedObjectSet'
    STREAM_VERSION = '0'

    def __init__(self, stream):

        reader = YamldataReader(stream, stream_type=self.STREAM_TYPE, stream_version=self.STREAM_VERSION)

##         print
##         print "yamldata:"
##         print "version", reader.version
##         print "stream_type", reader.stream_type
##         print "stream_version", reader.stream_version
##         print "stream_options", reader.stream_options
        
        stream_options = dict()
        if reader.stream_options is not None:
            stream_options.update(option.split('=') for option in reader.stream_options.split())
        # XXX verify option keys and values
##         print "stream_options:", stream_options

        def import_package(name):
            module = __import__(name)
            components = name.split('.')
            for component in components[1:]:
                module = getattr(module, component)
            return module

        textdata = list()

        requires_index = not stream_options.get('implicit_index')
        min_length = 3 if requires_index else 2

        object_set = self.object_set = list()

        for index, tokens in enumerate(reader):
            if len(tokens) < min_length:
                raise ValueError("expected at least %d tokens, got %s" % (min_length, ''.join(repr(token) for token in tokens),))
            if requires_index:
                token0 = tokens.pop(0)
                if token0 != str(index):
                    raise ValueError("expected index of %s, got %s: perhaps you need to renumber lines after editing, or you need 'implicit_index=True' in your stream_options" % (repr(str(index)), repr(token0),))

            module_name, factory_name, args = tokens[0], tokens[1], tokens[2:]
##             print module_name, factory_name, args

            mod = import_package(module_name)
##             print "mod:", mod
            factory = getattr(mod, factory_name)
##             print "factory:", factory
##             print "args:", args
            # here is where we create the object
            obj = factory(args)

            object_set.append(((module_name, factory_name, args,), obj,))

            textdata.append(tokens)

        self.textdata = tuple(textdata)
        
        object_set = tuple(object_set)
        
    def __iter__(self):
        return iter(self.object_set)
            
    def serialize(self, stream):
        writer = TextdataWriter(stream, stream_type=self.STREAM_TYPE, stream_version=self.STREAM_VERSION)
        for tokens in self.textdata:
            writer.write_tokens_newline(tokens)
        writer.close()



# Textdata version

from onyx.textdata.textdata import TextdataReader, TextdataWriter

class ObsoleteIndexedObjectSet(object):

    FILE_TYPE = 'IndexedObjectSet'
    FILE_VERSION = '0'

    def __init__(self, stream):

        reader = TextdataReader(stream, file_type=self.FILE_TYPE, file_version=self.FILE_VERSION)

        def import_package(name):
            module = __import__(name)
            components = name.split('.')
            for component in components[1:]:
                module = getattr(module, component)
            return module

        textdata = list()

        object_set = list()
        min_length = 3
        for index, tokens in enumerate(reader):
            if len(tokens) < min_length:
                raise ValueError("expected at least %d tokens, got %s" % (min_length, ''.join(repr(token) for token in tokens),))
            if tokens[0] != str(index):
                raise ValueError("expected index of %s, got %s" % (repr(str(index)), repr(tokens[0]),))

            module_name, factory_name, args = tokens[1], tokens[2], tokens[3:]
            print module_name, factory_name, args

            mod = import_package(module_name)
            print "mod:", mod
            factory = getattr(mod, factory_name)
            print "factory:", factory
            print "args:", args
            obj = factory(*args)

            object_set.append(((module_name, factory_name, args,), obj,))

            textdata.append(tokens)

        self.textdata = tuple(textdata)
        
        object_set = tuple(object_set)
        
            
    def serialize(self, stream):
        writer = TextdataWriter(stream, file_type=self.FILE_TYPE, file_version=self.FILE_VERSION)
        for tokens in self.textdata:
            writer.write_tokens_newline(tokens)
        writer.close()



if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
