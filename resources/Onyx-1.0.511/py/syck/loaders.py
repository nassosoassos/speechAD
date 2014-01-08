"""
syck.loaders is a high-level wrapper for the Syck YAML parser.
Do not use it directly, use the module 'syck' instead.
"""

# Python 2.2 compatibility
from __future__ import generators

try:
    import datetime
except ImportError:
    pass

try:
    Set = set
except:
    try:
        from sets import Set
    except ImportError:
        def Set(items):
            set = {}
            for items in items:
                set[items] = None
            return set

import _syck

import sys, re, warnings

__all__ = ['GenericLoader', 'Loader',
    'parse', 'load', 'parse_documents', 'load_documents',
    'NotUnicodeInputWarning']

class NotUnicodeInputWarning(UserWarning):
    pass

class GenericLoader(_syck.Parser):
    """
    GenericLoader constructs primitive Python objects from YAML documents.
    """

    def load(self):
        """
        Loads a YAML document from the source and return a native Python
        object. On EOF, returns None and set the eof attribute on.
        """
        node = self.parse()
        if self.eof:
            return
        return self._convert(node, {})

    def _convert(self, node, node_to_object):
        if node in node_to_object:
            return node_to_object[node]
        value = None
        if node.kind == 'scalar':
            value = node.value
        elif node.kind == 'seq':
            value = []
            for item_node in node.value:
                value.append(self._convert(item_node, node_to_object))
        elif node.kind == 'map':
            value = {}
            for key_node in node.value:
                key_object = self._convert(key_node, node_to_object)
                value_object = self._convert(node.value[key_node],
                        node_to_object)
                try:
                    if key_object in value:
                        value = None
                        break
                    value[key_object] = value_object
                except TypeError:
                    value = None
                    break
            if value is None:
                value = []
                for key_node in node.value:
                    key_object = self._convert(key_node, node_to_object)
                    value_object = self._convert(node.value[key_node],
                            node_to_object)
                value.append((key_object, value_object))
        node.value = value
        object = self.construct(node)
        node_to_object[node] = object
        return object

    def construct(self, node):
        """Constructs a Python object by the given node."""
        return node.value

class Merge:
    """Represents the merge key '<<'."""
    pass

class Default:
    """Represents the default key '='."""
    pass

class Loader(GenericLoader):
    """
    Loader constructs native Python objects from YAML documents.
    """

    inf_value = 1e300000
    nan_value = inf_value/inf_value

    timestamp_expr = re.compile(r'(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)'
            r'(?:'
                r'(?:[Tt]|[ \t]+)(?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)'
                r'(?:\.(?P<micro>\d+)?)?'
                r'[ \t]*(?:Z|(?P<zhour>[+-]\d\d)(?::(?P<zminute>\d\d))?)?'
            r')?')

    merge_key = Merge()
    default_key = Default()

    non_ascii = []
    for i in range(256):
        ch = chr(i)
        if ch.isalnum():
            non_ascii.append(ch)
        else:
            non_ascii.append('_')
    non_ascii = ''.join(non_ascii)

    python_bools = {'True': True, 'False': False}

    class python_class:
        pass

    def find_constructor(self, node):
        """
        Returns the contructor for generating a Python object for the given
        node.

        The node tags are mapped to constructors by the following rule:

        Tag                             Constructor
        ---                             -----------
        tag:yaml.org,2002:type          construct_type
        tag:python.yaml.org,2002:type   construct_python_type
        x-private:type                  construct_private_type
        tag:domain.tld,2002:type        construct_domain_tld_2002_type

        See the method code for more details.
        """
        parts = []
        if node.tag:
            parts = node.tag.split(':')
        if parts:
            if parts[0] == 'tag':
                parts.pop(0)
                if parts:
                    if parts[0] == 'yaml.org,2002':
                        parts.pop(0)
                    elif parts[0] == 'python.yaml.org,2002':
                        parts[0] = 'python'
            elif parts[0] == 'x-private':
                parts[0] = 'private'
        parts = [part.translate(self.non_ascii) for part in parts]
        while parts:
            method = 'construct_'+'_'.join(parts)
            if hasattr(self, method):
                return getattr(self, method)
            parts.pop()

    def construct(self, node):
        """Constructs a Python object by the given node."""
        if node.kind == 'map' and self.merge_key in node.value:
            self.merge_maps(node)
        constructor = self.find_constructor(node)
        if constructor:
            return constructor(node)
        else:
            return node.value

    def construct_null(self, node):
        return None

    def construct_bool_yes(self, node):
        return True

    def construct_bool_no(self, node):
        return False

    def construct_str(self, node):
        try:
            value = unicode(node.value, 'utf-8')
        except UnicodeDecodeError:
            warnings.warn("scalar value is not utf-8", NotUnicodeInputWarning)
            return node.value
        try:
            return value.encode('ascii')
        except UnicodeEncodeError:
            return value

    def construct_numeric_base60(self, num_type, node):
        digits = [num_type(part) for part in node.value.split(':')]
        digits.reverse()
        base = 1
        value = num_type(0)
        for digit in digits:
            value += digit*base
            base *= 60
        return value

    def construct_int(self, node):
        return int(node.value)

    def construct_int_hex(self, node):
        return int(node.value, 16)

    def construct_int_oct(self, node):
        return int(node.value, 8)

    def construct_int_base60(self, node):
        return self.construct_numeric_base60(int, node)

    def construct_float(self, node):
        return float(node.value)
    construct_float_fix = construct_float
    construct_float_exp = construct_float

    def construct_float_base60(self, node):
        return self.construct_numeric_base60(float, node)

    def construct_float_inf(self, node):
        return self.inf_value

    def construct_float_neginf(self, node):
        return -self.inf_value

    def construct_float_nan(self, node):
        return self.nan_value

    def construct_binary(self, node):
        return node.value.decode('base64')

    def construct_timestamp(self, node):
        match = self.timestamp_expr.match(node.value)
        values = match.groupdict()
        for key in values:
            if values[key]:
                values[key] = int(values[key])
            else:
                values[key] = 0
        micro = values['micro']
        if micro:
            while 10*micro < 1000000:
                micro *= 10
        stamp = datetime.datetime(values['year'], values['month'], values['day'],
                values['hour'], values['minute'], values['second'], micro)
        diff = datetime.timedelta(hours=values['zhour'], minutes=values['zminute'])
        return stamp-diff
    construct_timestamp_ymd = construct_timestamp
    construct_timestamp_iso8601 = construct_timestamp
    construct_timestamp_spaced = construct_timestamp

    def construct_merge(self, node):
        return self.merge_key

    def construct_default(self, node):
        return self.default_key

    def merge_maps(self, node):
        maps = node.value[self.merge_key]
        del node.value[self.merge_key]
        if not isinstance(maps, list):
            maps = [maps]
        maps.reverse()
        maps.append(node.value.copy())
        for item in maps:
            node.value.update(item)

    def construct_omap(self, node):
        omap = []
        for mapping in node.value:
            for key in mapping:
                omap.append((key, mapping[key]))
        return omap

    def construct_pairs(self, node): # Same as construct_omap.
        pairs = []
        for mapping in node.value:
            for key in mapping:
                pairs.append((key, mapping[key]))
        return pairs

    def construct_set(self, node):
        return Set(node.value)

    def construct_python_none(self, node):
        return None

    def construct_python_bool(self, node):
        return self.python_bools[node.value]

    def construct_python_int(self, node):
        return int(node.value)

    def construct_python_long(self, node):
        return long(node.value)

    def construct_python_float(self, node):
        return float(node.value)

    def construct_python_complex(self, node):
        return complex(node.value)

    def construct_python_str(self, node):
        return str(node.value)

    def construct_python_unicode(self, node):
        return unicode(node.value, 'utf-8')

    def construct_python_list(self, node):
        return node.value

    def construct_python_tuple(self, node):
        return tuple(node.value)

    def construct_python_dict(self, node):
        return node.value

    def find_python_object(self, node):
        full_name = node.tag.split(':')[3]
        parts = full_name.split('.')
        object_name = parts.pop()
        module_name = '.'.join(parts)
        if not module_name:
            module_name = '__builtin__'
        else:
            __import__(module_name)
        return getattr(sys.modules[module_name], object_name)

    def find_python_state(self, node):
        if node.kind == 'seq':
            args = node.value
            kwds = {}
            state = {}
        else:
            args = node.value.get('args', [])
            kwds = node.value.get('kwds', {})
            state = node.value.get('state', {})
        return args, kwds, state

    def set_python_state(self, object, state):
        if hasattr(object, '__setstate__'):
            object.__setstate__(state)
        else:
            slotstate = {}
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if hasattr(object, '__dict__'):
                object.__dict__.update(state)
            elif state:
                slotstate.update(state)
            for key, value in slotstate.items():
                setattr(object, key, value)

    def construct_python_name(self, node):
        return self.find_python_object(node)

    def construct_python_module(self, node):
        module_name = node.tag.split(':')[3]
        __import__(module_name)
        return sys.modules[module_name]

    def construct_python_object(self, node):
        cls = self.find_python_object(node)
        if type(cls) is type(self.python_class):
            if hasattr(cls, '__getnewargs__'):
                object = cls()
            else:
                object = self.python_class()
                object.__class__ = cls
        else:
            object = cls.__new__(cls)
        self.set_python_state(object, node.value)
        return object

    def construct_python_new(self, node):
        cls = self.find_python_object(node)
        args, kwds, state = self.find_python_state(node)
        if type(cls) is type(self.python_class):
            object = cls(*args, **kwds)
        else:
            object = cls.__new__(cls, *args, **kwds)
        self.set_python_state(object, state)
        return object

    def construct_python_apply(self, node):
        constructor = self.find_python_object(node)
        args, kwds, state = self.find_python_state(node)
        object = constructor(*args, **kwds)
        self.set_python_state(object, state)
        return object

def parse(source, Loader=Loader, **parameters):
    """Parses 'source' and returns the root of the 'Node' graph."""
    loader = Loader(source, **parameters)
    return loader.parse()

def load(source, Loader=Loader, **parameters):
    """Parses 'source' and returns the root object."""
    loader = Loader(source, **parameters)
    return loader.load()

def parse_documents(source, Loader=Loader, **parameters):
    """Iterates over 'source' and yields the root 'Node' for each document."""
    loader = Loader(source, **parameters)
    while True:
        node = loader.parse()
        if loader.eof:
            break
        yield node

def load_documents(source, Loader=Loader, **parameters):
    """Iterates over 'source' and yields the root object for each document."""
    loader = Loader(source, **parameters)
    while True:
        object = loader.load()
        if loader.eof:
            break
        yield object

